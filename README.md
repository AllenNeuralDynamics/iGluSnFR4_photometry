# iGluSnFR4 photometry

Analysis codes to reproduce panels in the fiber photometry figure (Figure 5) shown in the iGluSnFR4 paper (Aggarwal et al., 2025)

Preprint: https://www.biorxiv.org/content/10.1101/2025.03.20.643984v1

Publication: TBA


# For CodeOcean users

For source code, see this [github repo](https://github.com/AllenNeuralDynamics/iGluSnFR4_photometry) associated with the reproducible capsule.



# Experiments and data

The dataset involves fiber photometry measurements in behaving mice engaged in Pavlovian conditioning, where a water reward is delivered to mildly water-deprived mice following an auditory conditioned stimulus.
Behavioural task control and photometry data acquisition were performed using custom-written Bonsai software, as described in [the Github repo](https://github.com/AllenNeuralDynamics/PavlovianCond_Bonsai).

Structure of the fiber photometry data is described in the [AIND file-standard repo](https://github.com/AllenNeuralDynamics/aind-file-standards/blob/main/file_formats/fip.md).
Behavior data will be described soon in the same file-standard repo.

# Reproducing these results 

### Environment 

The environment for executing this script is defined in `/environment/Dockerfile`. You can either build this image yourself, or download our pre-built docker image. To do this, export the repository by navigating to AIND's [Public Collections](https://codeocean.allenneuraldynamics.org/collections/4a2d5da6-b053-43fe-9180-1912d787c59e), open this capsule, and export it from the menu (Capsule -> Export...). The downloaded repository will contain a `REPRODUCING.md` file with instructions you can follow.

### Data

The data used in this capsule are in a public S3 bucket at the following paths:

```
s3://aind-open-data/behavior_734805_2024-09-27_19-13-15
s3://aind-open-data/behavior_734806_2024-09-27_19-18-53
s3://aind-open-data/behavior_734808_2024-09-20_14-50-38
s3://aind-open-data/behavior_734809_2024-09-18_16-23-20
s3://aind-open-data/behavior_734811_2024-09-20_13-36-31
s3://aind-open-data/behavior_734812_2024-09-18_15-02-21
s3://aind-open-data/behavior_734813_2024-09-27_17-39-57
s3://aind-open-data/behavior_734814_2024-09-30_17-09-46
s3://aind-open-data/behavior_744066_2024-11-11_14-23-37
s3://aind-open-data/behavior_751752_2024-10-28_15-21-34
s3://aind-open-data/behavior_754977_2024-10-17_14-55-44
s3://aind-open-data/behavior_754979_2024-10-21_10-56-14
s3://aind-open-data/behavior_759856_2024-10-28_17-16-11
s3://aind-open-data/behavior_759857_2024-10-28_17-22-39
s3://aind-open-data/behavior_761754_2024-11-12_17-20-00
s3://aind-open-data/behavior_762321_2024-11-12_16-10-04
s3://aind-open-data/behavior_762327_2024-11-12_16-13-27
s3://aind-open-data/behavior_763856_2024-11-12_18-24-15
```

You can download them using standard S3 client tools like the [awscli](https://aws.amazon.com/cli/), [rclone](https://rclone.org/), and others. When running this capsule locally, data are expected to be in a `{repo-directory}/data/combined/` folder. 

### Code

Once you have a copy of the repository, environment, and data, run `bash {repository-directory}/code/run` inside the docker container. This triggers the following .py files in this order:
1. `/code/main.py`   
This step does signal preprocessing (using preprocessing functions organised in `/code/preprocessing.py`), PSTH preparation, saving intermediate data and figures (not shown in the paper) for each session.  
2. `/code/representativeplots_SFv4.py`   
This step produces the representative plot shown in Figure 5b and c, based on the intermediate PSTH data from the step1.
3. `/code/representativeplots_3v4.py`   
This step produces the representative plot shown in Figure 5b and c, based on the intermediate PSTH data from the step1.
4. `/code/summaryplots.py`   
This step produces the summary quantification shown in Figure 5d.

All figures used in the publication will be saved in the `/results/Fig_publication` folder with filenames corresponding to figure panels.

