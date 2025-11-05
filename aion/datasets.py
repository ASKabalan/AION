import os
import itertools
from abc import ABC, abstractmethod
from typing import Optional, Iterator, Union, Tuple
import torch
import numpy as np
from datasets import load_dataset, Dataset, load_from_disk
from tqdm import tqdm

from aion.modalities import HSCImage, DESISpectrum, EuclidImage


HSC_G_TO_EUCLID_VIS = (0.1312 , 0.01147)
HSC_2_TO_EUCLID_H   = (0.02096 , 0.005976)
HSC_Y_TO_EUCLID_J   = (0.03052 , -0.0003554)
HSC_R_TO_EUCLID_Y = (0.008414 , 0.01346)

class AIONDataset(ABC):
    def __init__(
        self,
        cache_dir: str,
        batch_size: int = 32,
        batch_count: Optional[int] = None,
        device: str = "cpu",
        split: str = "train",
    ):
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.device = device
        self.split = split

        self.slice_dir = os.path.join(cache_dir, f"{self._dataset_name()}_slice")
        self.samples = None
        self._ensure_samples()

    @abstractmethod
    def _dataset_name(self) -> str:
        pass

    @abstractmethod
    def _repo_id(self) -> str:
        pass

    @abstractmethod
    def _convert_sample(self, sample: dict):
        pass

    def _get_split(self) -> str:
        return self.split

    def _num_samples_needed(self) -> Optional[int]:
        if self.batch_count is None:
            return None
        return self.batch_size * self.batch_count

    def _ensure_samples(self):
        num_needed = self._num_samples_needed()

        if os.path.isdir(self.slice_dir):
            try:
                ds = load_from_disk(self.slice_dir)
            except (FileNotFoundError, Exception) as e:
                print(f"Warning: Cache directory exists but is invalid ({e}). Removing and re-creating...")
                import shutil
                shutil.rmtree(self.slice_dir)
                self._ensure_samples()
                return
            existing_count = len(ds)

            if num_needed is None or existing_count >= num_needed:
                self.samples = [ds[i] for i in range(existing_count if num_needed is None else num_needed)]
                print(f"Loaded {len(self.samples)} samples from disk: {self.slice_dir}")
                return

            print(f"Found {existing_count} cached samples, need {num_needed}. Streaming additional samples...")
            existing_samples = [ds[i] for i in range(existing_count)]
            additional_needed = num_needed - existing_count

            ds_stream = load_dataset(
                self._repo_id(),
                cache_dir=os.path.join(self.cache_dir, self._dataset_name()),
                split=self._get_split(),
                streaming=True,
            )

            new_samples = list(itertools.islice(
                itertools.islice(ds_stream, existing_count, None),
                additional_needed
            ))

            all_samples = existing_samples + new_samples
            Dataset.from_list(all_samples).save_to_disk(self.slice_dir)
            self.samples = all_samples
            print(f"Streamed {len(new_samples)} additional samples and saved to {self.slice_dir}")
        else:
            print(f"No cache found. Streaming samples from {self._repo_id()}...")
            ds_stream = load_dataset(
                self._repo_id(),
                cache_dir=os.path.join(self.cache_dir, self._dataset_name()),
                split=self._get_split(),
                streaming=True,
            )

            if num_needed is None:
                samples = list(tqdm(ds_stream, desc="Streaming all samples"))
            else:
                samples = list(itertools.islice(
                    tqdm(ds_stream, total=num_needed, desc="Streaming samples"),
                    num_needed
                ))

            Dataset.from_list(samples).save_to_disk(self.slice_dir)
            self.samples = samples
            print(f"Streamed and saved {len(samples)} samples to {self.slice_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= (len(self.samples) + self.batch_size - 1) // self.batch_size:
            raise IndexError(f"Batch index {idx} out of range")
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.samples))
        batch_samples = self.samples[start_idx:end_idx]
        return self._convert_batch(batch_samples)

    def __iter__(self) -> Iterator:
        for i in range(0, len(self.samples), self.batch_size):
            batch_samples = self.samples[i:i + self.batch_size]
            yield self._convert_batch(batch_samples)

    def _convert_batch(self, batch_samples):
        converted = [self._convert_sample(sample) for sample in batch_samples]

        if isinstance(converted[0], tuple):
            return tuple(self._stack_modalities([c[i] for c in converted]) for i in range(len(converted[0])))
        else:
            return self._stack_modalities(converted)

    def _stack_modalities(self, modalities):
        if len(modalities) == 0:
            return None

        first = modalities[0]

        if isinstance(first, HSCImage) or isinstance(first, EuclidImage):
            flux_list = [m.flux for m in modalities]
            flux = torch.stack(flux_list).to(self.device)
            return type(first)(flux=flux, bands=first.bands)

        elif isinstance(first, DESISpectrum):
            flux_list = [m.flux for m in modalities]
            ivar_list = [m.ivar for m in modalities]
            wavelength_list = [m.wavelength for m in modalities]
            mask_list = [m.mask for m in modalities]

            flux = torch.stack(flux_list).to(self.device)
            ivar = torch.stack(ivar_list).to(self.device)
            wavelength = torch.stack(wavelength_list).to(self.device)
            mask = torch.stack(mask_list).to(self.device)

            return DESISpectrum(
                flux=flux,
                ivar=ivar,
                wavelength=wavelength,
                mask=mask,
            )

        else:
            raise ValueError(f"Unknown modality type: {type(first)}")


class HSCDataset(AIONDataset):
    def _dataset_name(self) -> str:
        return "hsc"

    def _repo_id(self) -> str:
        return "MultimodalUniverse/hsc"

    def _convert_sample(self, sample: dict) -> HSCImage:
        flux = torch.tensor(np.array(sample['image']['flux']), dtype=torch.float32)
        bands = [band.upper() for band in sample['image']['band']]
        return HSCImage(flux=flux, bands=bands)


class DESIDataset(AIONDataset):
    def _dataset_name(self) -> str:
        return "desi"

    def _repo_id(self) -> str:
        return "MultimodalUniverse/desi"

    def _convert_sample(self, sample: dict) -> DESISpectrum:
        spectrum = sample['spectrum']

        flux = torch.tensor(np.array(spectrum['flux']), dtype=torch.float32)
        ivar = torch.tensor(np.array(spectrum['ivar']), dtype=torch.float32)
        wavelength = torch.tensor(np.array(spectrum['lambda']), dtype=torch.float32)
        mask = torch.tensor(np.array(spectrum['mask']), dtype=torch.bool)

        return DESISpectrum(
            flux=flux,
            ivar=ivar,
            wavelength=wavelength,
            mask=mask,
        )


class EuclidDataset(AIONDataset):
    def _dataset_name(self) -> str:
        return "euclid"

    def _repo_id(self) -> str:
        return "msiudek/astroPT_euclid_dataset"

    def _convert_sample(self, sample: dict) -> EuclidImage:
        vis_image = np.array(sample['VIS_image'])
        nisp_y_image = np.array(sample['NISP_Y_image'])
        nisp_j_image = np.array(sample['NISP_J_image'])
        nisp_h_image = np.array(sample['NISP_H_image'])

        flux = torch.tensor(
            np.stack([vis_image, nisp_y_image, nisp_j_image, nisp_h_image], axis=0),
            dtype=torch.float32
        )

        bands = ['EUCLID-VIS', 'EUCLID-Y', 'EUCLID-J', 'EUCLID-H']

        return EuclidImage(flux=flux, bands=bands)


class EuclidDESIDataset(AIONDataset):
    def _dataset_name(self) -> str:
        return "euclid_desi"

    def _repo_id(self) -> str:
        return "msiudek/astroPT_euclid_desi_dataset"

    def _get_split(self) -> str:
        return "train_batch_1"

    def _convert_sample(self, sample: dict) -> Tuple[HSCImage, DESISpectrum]:
        vis_image = np.array(sample['VIS_image']) * HSC_G_TO_EUCLID_VIS[0] + HSC_G_TO_EUCLID_VIS[1]
        nisp_y_image = np.array(sample['NISP_Y_image']) * HSC_R_TO_EUCLID_Y[0] + HSC_R_TO_EUCLID_Y[1]
        nisp_j_image = np.array(sample['NISP_J_image']) * HSC_Y_TO_EUCLID_J[0] + HSC_Y_TO_EUCLID_J[1]
        nisp_h_image = np.array(sample['NISP_H_image']) * HSC_2_TO_EUCLID_H[0] + HSC_2_TO_EUCLID_H[1]

        euclid_flux = torch.tensor(
            np.stack([vis_image, nisp_y_image, nisp_j_image, nisp_h_image], axis=0),
            dtype=torch.float32
        )

        euclid_as_hsc = HSCImage(
            flux=euclid_flux,
            bands=['EUCLID-VIS', 'EUCLID-Y', 'EUCLID-J', 'EUCLID-H']
            #bands=['HSC-VIS', 'HSC-Y', 'HSC-J', 'HSC-H']
        )

        spectrum = sample['spectrum']
        flux = np.array(spectrum['flux'])
        error = np.array(spectrum['error'])
        wavelength = np.array(spectrum['wavelength'])

        ivar = np.where(error > 0, 1.0 / (error ** 2), 0.0)
        mask = error > 0

        desi_spectrum = DESISpectrum(
            flux=torch.tensor(flux, dtype=torch.float32),
            ivar=torch.tensor(ivar, dtype=torch.float32),
            wavelength=torch.tensor(wavelength, dtype=torch.float32),
            mask=torch.tensor(mask, dtype=torch.bool),
        )

        return euclid_as_hsc, desi_spectrum
