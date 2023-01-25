import os
import h5py
import torch
import numpy as np

from torch.utils.data import Dataset


def to_chunks(lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]


def window_slicer(data, width, step_size):
    return np.dstack([data[i:1+i-width or None:round(step_size)] for i in range(0,width)]).squeeze()


class DataAdapter(Dataset):
    def __init__(
        self,
        noise,
        eye_input,
        eye_target,
        joint_angle,
        angular_velocity,
        data_file,
        **kwargs
    ):

        self.noise = noise
        self.eye_input = eye_input
        self.eye_target = eye_target
        self.joint_angle = joint_angle
        self.angular_velocity = angular_velocity
        self.data_file = data_file

    def __len__(self):
        return len(self.eye_target)

    def __getitem__(self, idx):
        noise = self.noise[idx]
        eye_input = self.eye_input[idx]
        eye_target = self.eye_target[idx]
        joint_angle = self.joint_angle[idx]
        angular_velocity = self.angular_velocity[idx]

        noise = self.data_file[noise][()].astype(np.float32)
        eye_input = self.data_file[eye_input][()].astype(np.float32)
        eye_target = self.data_file[eye_target][()].astype(np.float32)
        joint_angle = self.data_file[joint_angle][()].astype(np.float32)
        angular_velocity = self.data_file[angular_velocity][()].astype(np.float32)

        noise = torch.from_numpy(noise)
        eye_input = torch.from_numpy(eye_input)
        eye_target = torch.from_numpy(eye_target)
        joint_angle = torch.from_numpy(joint_angle)
        angular_velocity = torch.from_numpy(angular_velocity)

        return (
            (
                eye_input,
                joint_angle,
                angular_velocity,
                noise,
            ),
            eye_target
        )


class ParallelBatchDataAdapter(Dataset):
    def __init__(
        self,
        noise,
        eye_input,
        eye_target,
        joint_angle,
        angular_velocity,
        data_file,
        batch_size,
        block_size,
        **kwargs
    ):
        self.noise = noise
        self.eye_input = eye_input
        self.eye_target = eye_target
        self.joint_angle = joint_angle
        self.angular_velocity = angular_velocity
        self.data_file = data_file
        self.batch_size = batch_size
        self.block_size = min(block_size, len(self.eye_target))

    def __len__(self):
        return self.block_size

    def __getitem__(self, idx):
        noise = self.noise[idx: idx + self.batch_size].ravel()
        eye_input = self.eye_input[idx: idx + self.batch_size].ravel()
        eye_target = self.eye_target[idx: idx + self.batch_size].ravel()
        joint_angle = self.joint_angle[idx: idx + self.batch_size].ravel()
        angular_velocity = self.angular_velocity[idx: idx + self.batch_size].ravel()

        noise = np.stack([self.data_file[ref][()] for ref in noise]).astype(np.float32)
        eye_input = np.stack([self.data_file[ref][()] for ref in eye_input]).astype(np.float32)
        eye_target = np.stack([self.data_file[ref][()] for ref in eye_target]).astype(np.float32)
        joint_angle = np.stack([self.data_file[ref][()] for ref in joint_angle]).astype(np.float32)
        angular_velocity = np.stack([self.data_file[ref][()] for ref in angular_velocity]).astype(np.float32)

        noise = torch.from_numpy(noise)
        eye_input = torch.from_numpy(eye_input)
        eye_target = torch.from_numpy(eye_target)
        joint_angle = torch.from_numpy(joint_angle)
        angular_velocity = torch.from_numpy(angular_velocity)

        return (
            (
                eye_input,
                joint_angle,
                angular_velocity,
                noise,
            ),
            eye_target
        )


class RollingDataAdapter:

    def __init__(
        self,
        noise,
        eye_input,
        eye_target,
        joint_angle,
        angular_velocity,
        data_file,
        batch_size,
        block_size
    ):
        self.noise = noise
        self.eye_input = eye_input
        self.eye_target = eye_target
        self.joint_angle = joint_angle
        self.angular_velocity = angular_velocity
        self.data_file = data_file
        self.batch_size = batch_size
        self.block_size = block_size

        max_length = len(noise)
        raw_idxs = np.arange(max_length)
        if max_length < block_size + batch_size:
            self.slices = [raw_idxs, ]
            self.batch_size = 1
        else:
            self.slices = window_slicer(raw_idxs, block_size + batch_size, batch_size)
    
    def __len__(self):
        return len(self.slices)

    def __iter__(self):
        for rolling_slice in self.slices:
            rolling_noise = self.noise[rolling_slice]
            rolling_eye_input = self.eye_input[rolling_slice]
            rolling_eye_target = self.eye_target[rolling_slice]
            rolling_joint_angle = self.joint_angle[rolling_slice]
            rolling_angular_velocity = self.angular_velocity[rolling_slice]

            yield ParallelBatchDataAdapter(
                rolling_noise,
                rolling_eye_input,
                rolling_eye_target,
                rolling_joint_angle,
                rolling_angular_velocity,
                self.data_file,
                self.batch_size,
                self.block_size
            )


class RollingBlock:
    def __init__(
        self,
        noise,
        eye_input,
        eye_target,
        joint_angle,
        angular_velocity,
        data_file,
        batch_size,
        block_size,
        buffer_zone,
        closed_loop_block_size
    ):
        self.noise = noise
        self.eye_input = eye_input
        self.eye_target = eye_target
        self.joint_angle = joint_angle
        self.angular_velocity = angular_velocity
        self.data_file = data_file
        self.block_size = block_size
        self.batch_size = batch_size
        self.closed_loop_block_size = closed_loop_block_size

        if isinstance(buffer_zone, int):
            buffer_zone = (-buffer_zone, buffer_zone)
        elif isinstance(buffer_zone, (tuple, list)):
            if len(buffer_zone) == 1:
                buffer_zone = (-buffer_zone[0], buffer_zone[0])
            else:
                raise ValueError(f"Provided buffer_zone {buffer_zone} has length more than 2")
        elif isinstance(buffer_zone, np.ndarray):
            if buffer_zone.ndim != 1:
                raise ValueError(f"Provided buffer_zone {buffer_zone} has ndim more than 1")
            else:
                buffer_zone = (-buffer_zone[0], buffer_zone[0])

        self.buffer_zone = np.arange(buffer_zone[0], buffer_zone[1])

        self.raw_idxs = np.arange(len(noise))

        self.chunks = to_chunks(self.raw_idxs, block_size + (2 * buffer_zone[1]))

    def __len__(self):
        return len(self.chunks)

    def __iter__(self):
        for test_chunk in self.chunks:
            train_chunk = self.raw_idxs.copy()
            train_chunk = np.delete(train_chunk, test_chunk)

            train_noise = self.noise[train_chunk]
            train_eye_input = self.eye_input[train_chunk]
            train_eye_target = self.eye_target[train_chunk]
            train_joint_angle = self.joint_angle[train_chunk]
            train_angular_velocity = self.angular_velocity[train_chunk]
            
            train_data = DataAdapter(
                train_noise,
                train_eye_input,
                train_eye_target,
                train_joint_angle,
                train_angular_velocity,
                self.data_file
            )
            
            test_chunk = np.delete(test_chunk, self.buffer_zone)
            
            test_noise = self.noise[test_chunk]
            test_eye_input = self.eye_input[test_chunk]
            test_eye_target = self.eye_target[test_chunk]
            test_joint_angle = self.joint_angle[test_chunk]
            test_angular_velocity = self.angular_velocity[test_chunk]
            
            test_data = DataAdapter(
                test_noise,
                test_eye_input,
                test_eye_target,
                test_joint_angle,
                test_angular_velocity,
                self.data_file
            )

            closed_loop_data = RollingDataAdapter(
                test_noise,
                test_eye_input,
                test_eye_target,
                test_joint_angle,
                test_angular_velocity,
                self.data_file,
                self.batch_size,
                self.closed_loop_block_size
            )

            yield train_data, test_data, closed_loop_data


class Dataset:
    def __init__(
        self,
        mode,
        test_keys,
        data_file,
        batch_size,
        block_size,
        buffer_zone,
        closed_loop_block_size
    ):
        self.mode = mode
        self.test_keys = test_keys
        self.data_file = data_file
        self.batch_size = batch_size
        self.block_size = block_size
        self.buffer_zone = buffer_zone
        self.closed_loop_block_size = closed_loop_block_size

    def __len__(self):
        return len(self.test_keys)

    def __iter__(self):
        with h5py.File(self.data_file, "r") as data_file:
            for test_key in self.test_keys:
                if self.mode == "cross_subject":
                    train_noise = data_file[self.mode][test_key]["train"]["noise"][()]
                    train_eye_input = data_file[self.mode][test_key]["train"]["eyeinput"][()]
                    train_eye_target = data_file[self.mode][test_key]["train"]["eyetarget"][()]
                    train_joint_angle = data_file[self.mode][test_key]["train"]["jointangle"][()]
                    train_angular_velocity = data_file[self.mode][test_key]["train"]["angularVelocity"][()]
                    
                    train_data = DataAdapter(
                        train_noise,
                        train_eye_input,
                        train_eye_target,
                        train_joint_angle,
                        train_angular_velocity,
                        data_file
                    )
                    
                    test_noise = data_file[self.mode][test_key]["test"]["noise"][()]
                    test_eye_input = data_file[self.mode][test_key]["test"]["eyeinput"][()]
                    test_eye_target = data_file[self.mode][test_key]["test"]["eyetarget"][()]
                    test_joint_angle = data_file[self.mode][test_key]["test"]["jointangle"][()]
                    test_angular_velocity = data_file[self.mode][test_key]["test"]["angularVelocity"][()]
                    
                    test_data = DataAdapter(
                        test_noise,
                        test_eye_input,
                        test_eye_target,
                        test_joint_angle,
                        test_angular_velocity,
                        data_file
                    )

                    closed_loop_data = RollingDataAdapter(
                        test_noise,
                        test_eye_input,
                        test_eye_target,
                        test_joint_angle,
                        test_angular_velocity,
                        data_file,
                        self.batch_size,
                        self.block_size
                    )

                    dataset = ((train_data, test_data, closed_loop_data), )

                else:
                    train_noise = data_file[self.mode][test_key]["train"]["noise"][()]
                    train_eye_input = data_file[self.mode][test_key]["train"]["eyeinput"][()]
                    train_eye_target = data_file[self.mode][test_key]["train"]["eyetarget"][()]
                    train_joint_angle = data_file[self.mode][test_key]["train"]["jointangle"][()]
                    train_angular_velocity = data_file[self.mode][test_key]["train"]["angularVelocity"][()]
                    
                    dataset = RollingBlock(
                        train_noise,
                        train_eye_input,
                        train_eye_target,
                        train_joint_angle,
                        train_angular_velocity,
                        data_file,
                        self.batch_size,
                        self.block_size,
                        self.buffer_zone,
                        self.closed_loop_block_size
                    )

                yield test_key, dataset
                    # Need rewrite for more advanced method of feeding data
        # return test_key, train_data, test_data, closed_loop_data


def get_dataset(folder, suffix, block_size, closed_loop_block_size, mode="cross_subject", batch_size=32, buffer_zone=300):
    data_file = os.path.join(
        folder,
        suffix,
        "dataset.h5")
    with h5py.File(data_file, "r") as _file:
        test_keys = list(_file[mode].keys())
    return Dataset(mode, test_keys, data_file, batch_size, block_size, buffer_zone, closed_loop_block_size)


class TimeSeriesDataset(Dataset):

    def __init__(self, dataset, mode):
        super().__init__()
        self.dataset = dataset[mode]
    
    def __len__(self):
        return self.dataset['length'][()].squeeze()
        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        eye_input_ref = self.dataset['eyeinput'][idx]
        joint_angle_ref = self.dataset['jointangle'][idx]
        angular_velocity_ref = self.dataset['angularVelocity'][idx]
        noise_ref = self.dataset['noise'][idx]
        eye_target_ref = self.dataset['eyetarget'][idx]
        return ((torch.from_numpy(self.dataset[eye_input_ref][()]),
                 torch.from_numpy(self.dataset[joint_angle_ref][()]),
                 torch.from_numpy(self.dataset[angular_velocity_ref][()]),
                 torch.from_numpy(self.dataset[noise_ref][()])),
                 torch.from_numpy(self.dataset[eye_target_ref][()])
                 )