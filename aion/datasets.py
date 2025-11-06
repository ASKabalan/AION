import os
import itertools
from abc import ABC, abstractmethod
from typing import Optional, Iterator, Union, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, Dataset, load_from_disk
from tqdm import tqdm

from aion.modalities import HSCImage, DESISpectrum, EuclidImage , Z


def collate_modalities(batch):
    if len(batch) == 0:
        return batch

    first = batch[0]

    if isinstance(first, tuple):
        return tuple(collate_modalities([item[i] for item in batch]) for i in range(len(first)))

    if isinstance(first, torch.Tensor):
        return torch.stack(batch)

    if isinstance(first, HSCImage) or isinstance(first, EuclidImage):
        flux_list = [item.flux for item in batch]
        flux = torch.stack(flux_list)
        return type(first)(flux=flux, bands=first.bands)

    if isinstance(first, DESISpectrum):
        flux_list = [item.flux for item in batch]
        ivar_list = [item.ivar for item in batch]
        wavelength_list = [item.wavelength for item in batch]
        mask_list = [item.mask for item in batch]

        flux = torch.stack(flux_list)
        ivar = torch.stack(ivar_list)
        wavelength = torch.stack(wavelength_list)
        mask = torch.stack(mask_list)

        return DESISpectrum(flux=flux, ivar=ivar, wavelength=wavelength, mask=mask)

    if isinstance(first, Z):
        value_list = [item.value for item in batch]
        value = torch.stack(value_list)
        return Z(value=value)

    return batch


HSC_G_TO_EUCLID_VIS = (0.1312 , 0.01147)
HSC_2_TO_EUCLID_H   = (0.02096 , 0.005976)
HSC_Y_TO_EUCLID_J   = (0.03052 , -0.0003554)
HSC_R_TO_EUCLID_Y = (0.008414 , 0.01346)

class AIONDataset(ABC):
    def __init__(
        self,
        cache_dir: str,
        max_entries: Optional[int] = None,
        device: str = "cpu",
        split: str = "train",
        mode: str = "streaming",
    ):
        self.cache_dir = cache_dir
        self.max_entries = max_entries
        self.device = device
        self.split = split
        self.mode = mode

        if mode not in ["streaming", "local"]:
            raise ValueError(f"mode must be 'streaming' or 'local', got '{mode}'")

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

    def _ensure_samples(self):
        if self.mode == "local":
            if not os.path.isdir(self.slice_dir):
                raise FileNotFoundError(
                    f"Local mode requires existing cache at {self.slice_dir}. "
                    f"Use mode='streaming' first to download the dataset."
                )
            try:
                ds = load_from_disk(self.slice_dir)
                self.samples = [ds[i] for i in range(len(ds))]
                print(f"Loaded {len(self.samples)} samples from local cache: {self.slice_dir}")
            except Exception as e:
                raise RuntimeError(f"Failed to load local cache from {self.slice_dir}: {e}")
            return

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

            if self.max_entries is None or existing_count >= self.max_entries:
                self.samples = [ds[i] for i in range(existing_count if self.max_entries is None else self.max_entries)]
                print(f"Loaded {len(self.samples)} samples from disk: {self.slice_dir}")
                return

            print(f"Found {existing_count} cached samples, need {self.max_entries}. Streaming additional samples...")
            existing_samples = [ds[i] for i in range(existing_count)]
            additional_needed = self.max_entries - existing_count

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

            if self.max_entries is None:
                samples = list(tqdm(ds_stream, desc="Streaming all samples"))
            else:
                samples = list(itertools.islice(
                    tqdm(ds_stream, total=self.max_entries, desc="Streaming samples"),
                    self.max_entries
                ))

            Dataset.from_list(samples).save_to_disk(self.slice_dir)
            self.samples = samples
            print(f"Streamed and saved {len(samples)} samples to {self.slice_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range")
        return self._convert_sample(self.samples[idx])

    def __iter__(self) -> Iterator:
        for sample in self.samples:
            yield self._convert_sample(sample)


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
        return "msiudek/astroPT_euclid_training_dataset"

    def _convert_sample(self, sample: dict) -> EuclidImage:
        vis_image = np.array(sample['VIS_image'])
        nisp_y_image = np.array(sample['NISP_Y_image'])
        nisp_j_image = np.array(sample['NISP_J_image'])
        nisp_h_image = np.array(sample['NISP_H_image'])

        flux = torch.tensor(
            np.stack([vis_image, nisp_y_image, nisp_j_image, nisp_h_image], axis=0),
            dtype=torch.float32
        )

        flux = F.interpolate(
            flux.unsqueeze(0),
            size=(160, 160),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        bands = ['EUCLID-VIS', 'EUCLID-Y', 'EUCLID-J', 'EUCLID-H']

        return EuclidImage(flux=flux, bands=bands)


class EuclidDESIDataset(AIONDataset):
    def _dataset_name(self) -> str:
        return "euclid_desi"

    def _repo_id(self) -> str:
        return "msiudek/astroPT_euclid_Q1_desi_dr1_dataset"

    def _get_split(self) -> str:
        return "train"

    def _convert_sample(self, sample: dict) -> Tuple[torch.Tensor, EuclidImage, DESISpectrum, Z]:
        vis_image = np.array(sample['VIS_image']) * HSC_G_TO_EUCLID_VIS[0] + HSC_G_TO_EUCLID_VIS[1]
        nisp_y_image = np.array(sample['NISP_Y_image']) * HSC_R_TO_EUCLID_Y[0] + HSC_R_TO_EUCLID_Y[1]
        nisp_j_image = np.array(sample['NISP_J_image']) * HSC_Y_TO_EUCLID_J[0] + HSC_Y_TO_EUCLID_J[1]
        nisp_h_image = np.array(sample['NISP_H_image']) * HSC_2_TO_EUCLID_H[0] + HSC_2_TO_EUCLID_H[1]
        redshift = sample['redshift']

        euclid_flux = torch.tensor(
            np.stack([vis_image, nisp_y_image, nisp_j_image, nisp_h_image], axis=0),
            dtype=torch.float32
        )

        euclid_flux = F.interpolate(
            euclid_flux.unsqueeze(0),
            size=(160, 160),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        euclid_image = EuclidImage(
            flux=euclid_flux,
            bands=['EUCLID-VIS', 'EUCLID-Y', 'EUCLID-J', 'EUCLID-H']
        )

        rgb_image = torch.tensor(np.array(sample['RGB_image']), dtype=torch.uint8)
        #if rgb_image.ndim == 3:
        #    rgb_image = rgb_image.permute(2, 0, 1)

        spectrum = sample['spectrum']
        flux = np.array(spectrum['flux'])
        error = np.array(spectrum['error'])
        wavelength = np.array(spectrum['wavelength'])

        ivar = np.where(error > 0, 1.0 / (error ** 2), 0.0)
        mask = torch.zeros_like(torch.tensor(flux, dtype=torch.float32), dtype=torch.bool)

        desi_spectrum = DESISpectrum(
            flux=torch.tensor(flux, dtype=torch.float32),
            ivar=torch.tensor(ivar, dtype=torch.float32),
            wavelength=torch.tensor(wavelength, dtype=torch.float32),
            mask=torch.tensor(mask, dtype=torch.bool),
        )

        redshift_modality = Z(value=torch.tensor([redshift], dtype=torch.float32))

        return rgb_image, euclid_image, desi_spectrum, redshift_modality
