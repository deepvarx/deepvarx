import os
import sys
import numpy as np
import h5py
import logging
from glob import glob
from tqdm import tqdm
from scipy import io as sio
import hdf5storage
from absl import flags
from absl import app
import math


FLAGS = flags.FLAGS

flags.DEFINE_integer('stepSize', 1,
                        """Number of steps to move.""")            
flags.DEFINE_integer('lags', 60,
                        """Size of lag window on which to predict.""")
flags.DEFINE_integer('predictionHorizon', 1,
                        """Steps ahead to predict.""")
flags.DEFINE_string('suffix', "",
                        """Suffix""")
flags.DEFINE_integer('sliceGap', 1, 
                        "Size of step gap between window slices")
flags.DEFINE_string('input', "/path/to/folder/*.mat", 
                             "Input location of MAT files")
flags.DEFINE_string('output', "/path/to/folder/prepped", 
                              "Output location of prepped data")

def windowSlicer(data, stepSize, width):
    return np.swapaxes(np.dstack([data[i:1+i-width or None:round(stepSize)] for i in range(0,width)]), 1, 2)


class DataPrep():
    def __init__(self, globpath):
        self.filelist = glob(globpath)
        folderextension = 'Lags{}StepSize{}PredictionHorizon{}'.format(FLAGS.lags,
                                                                       FLAGS.stepSize,
                                                                       FLAGS.predictionHorizon)

        folderextension+=FLAGS.suffix
        self.folder = os.path.join(FLAGS.output, folderextension)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        logging.info(f"FOLDER {self.folder} CREATED")

    #a wrapper function for the __readFiles function below
    def readFiles(self):
        self.__readFiles(self.filelist, FLAGS.lags, FLAGS.stepSize, FLAGS.predictionHorizon)


    def __readFiles(self, filelist, lags, stepSize, predictionHorizon):

        print("READING DATA...")
        _file = h5py.File(os.path.join(self.folder, 'dataset.h5'), 'w')

        dataset = {}
        files_bar = tqdm(total=len(filelist))
        for file in filelist:
            datasetFile = file.split('.')[0].split(os.sep)[-1]
            files_bar.set_description(f'Adding in-subject {datasetFile} ...')
            tempData = hdf5storage.loadmat(file)
            tempEyeinput = tempData['gaze'][:-predictionHorizon, :].astype(np.float32)
            tempEyetarget = tempData['gaze'][predictionHorizon:, :].astype(np.float32)
            tempJointAngle = tempData['jointAngle'][:-predictionHorizon, :].astype(np.float32)
            tempAngularVelocity = tempData['angularVelocity'][:-predictionHorizon, :].astype(np.float32)
            slicedEyeinput =  windowSlicer(tempEyeinput, stepSize, lags)
            slicedEyetarget =  windowSlicer(tempEyetarget, stepSize, lags)
            slicedJointAngle = windowSlicer(tempJointAngle, stepSize, lags)
            slicedAngularVelocity = windowSlicer(tempAngularVelocity, stepSize, lags)
            noise = np.random.normal(size=tempEyeinput.shape).astype(np.float32)
            slicedNoise = windowSlicer(noise, 1, FLAGS.lags)
            
            eye_input_refs = []
            eye_target_refs = []
            joint_angle_refs = []
            angular_velocity_refs = []
            noise_refs = []
            for idx, (sliced_eye_input, sliced_eye_target, sliced_joint_angle, sliced_angular_velocity, sliced_noise) in enumerate(
                zip(
                    slicedEyeinput,
                    slicedEyetarget[:,-1,:],
                    slicedJointAngle,
                    slicedAngularVelocity,
                    slicedNoise
                )
            ):
                raw_eye_input = _file.create_dataset(f"_raw_data/{datasetFile}/eyeinput/{idx}", data=sliced_eye_input)
                eye_input_refs.append(raw_eye_input.ref)

                raw_eye_target = _file.create_dataset(f"_raw_data/{datasetFile}/eyetarget/{idx}", data=sliced_eye_target)
                eye_target_refs.append(raw_eye_target.ref)

                raw_joint_angle = _file.create_dataset(f"_raw_data/{datasetFile}/jointangle/{idx}", data=sliced_joint_angle)
                joint_angle_refs.append(raw_joint_angle.ref)
                
                raw_angular_velocity = _file.create_dataset(f"_raw_data/{datasetFile}/angularVelocity/{idx}", data=sliced_angular_velocity)
                angular_velocity_refs.append(raw_angular_velocity.ref)

                raw_noise = _file.create_dataset(f"_raw_data/{datasetFile}/noise/{idx}", data=sliced_noise)
                noise_refs.append(raw_noise.ref)

            
            dataset[datasetFile] = {
                "eye_input_refs": eye_input_refs,
                "eye_target_refs": eye_target_refs,
                "joint_angle_refs": joint_angle_refs,
                "angular_velocity_refs": angular_velocity_refs,
                "noise_refs": noise_refs,
            }

            trainindex = [i for i in range(len(eye_input_refs))]
            train_eye_input_refs = [eye_input_refs[idx] for idx in trainindex]
            train_eye_target_refs = [eye_target_refs[idx] for idx in trainindex]
            train_joint_angle_refs = [joint_angle_refs[idx] for idx in trainindex]
            train_angular_velocity_refs = [angular_velocity_refs[idx] for idx in trainindex]
            train_noise_refs = [noise_refs[idx] for idx in trainindex]

            _file.create_dataset(os.path.join('in_subject', f'{datasetFile}', 'train', 'eyeinput'), data=train_eye_input_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('in_subject', f'{datasetFile}', 'train', 'eyetarget'), data=train_eye_target_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('in_subject', f'{datasetFile}', 'train', 'jointangle'), data=train_joint_angle_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('in_subject', f'{datasetFile}', 'train', 'angularVelocity'), data=train_angular_velocity_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('in_subject', f'{datasetFile}', 'train', 'noise'), data=train_noise_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('in_subject', f'{datasetFile}', 'train', 'length'), data=len(trainindex))
            files_bar.update(1)

            
        # cross_subject
        files_bar = tqdm(total=len(filelist))
        for file in filelist:
            datasetFile = file.split('.')[0].split(os.sep)[-1]
            files_bar.set_description(f'Adding cross-subject {datasetFile} ...')
            test_eye_input_refs = dataset[datasetFile]["eye_input_refs"]
            test_eye_target_refs = dataset[datasetFile]["eye_target_refs"]
            test_joint_angle_refs = dataset[datasetFile]["joint_angle_refs"]
            test_angular_velocity_refs = dataset[datasetFile]["angular_velocity_refs"]
            test_noise_refs = dataset[datasetFile]["noise_refs"]

            _file.create_dataset(os.path.join('cross_subject', f'{datasetFile}', 'test', 'eyeinput'), data=test_eye_input_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('cross_subject', f'{datasetFile}', 'test', 'eyetarget'), data=test_eye_target_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('cross_subject', f'{datasetFile}', 'test', 'jointangle'), data=test_joint_angle_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('cross_subject', f'{datasetFile}', 'test', 'angularVelocity'), data=test_angular_velocity_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('cross_subject', f'{datasetFile}', 'test', 'noise'), data=test_noise_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('cross_subject', f'{datasetFile}', 'test', 'length'), data=len(test_eye_input_refs))
            
            train_eye_input_refs = [ref for key in dataset.keys() if key not in [datasetFile] for ref in dataset[key]["eye_input_refs"]]
            train_eye_target_refs = [ref for key in dataset.keys() if key not in [datasetFile] for ref in dataset[key]["eye_target_refs"]]
            train_joint_angle_refs = [ref for key in dataset.keys() if key not in [datasetFile] for ref in dataset[key]["joint_angle_refs"]]
            train_angular_velocity_refs = [ref for key in dataset.keys() if key not in [datasetFile] for ref in dataset[key]["angular_velocity_refs"]]
            train_noise_refs = [ref for key in dataset.keys() if key not in [datasetFile] for ref in dataset[key]["noise_refs"]]
            
            _file.create_dataset(os.path.join('cross_subject', f'{datasetFile}', 'train', 'eyeinput'), data=train_eye_input_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('cross_subject', f'{datasetFile}', 'train', 'eyetarget'), data=train_eye_target_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('cross_subject', f'{datasetFile}', 'train', 'jointangle'), data=train_joint_angle_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('cross_subject', f'{datasetFile}', 'train', 'angularVelocity'), data=train_angular_velocity_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('cross_subject', f'{datasetFile}', 'train', 'noise'), data=train_noise_refs, dtype=h5py.ref_dtype)
            _file.create_dataset(os.path.join('cross_subject', f'{datasetFile}', 'train', 'length'), data=len(train_eye_input_refs))
            files_bar.update(1)
                
        _file.close()


def main(argv):
    dataprep = DataPrep(FLAGS.input)
    dataprep.readFiles()

if __name__ =="__main__":
    app.run(main)