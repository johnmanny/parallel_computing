# Appointment Prediction
The purpose of this project was to train a neural network for predicting whether
a patient would no-show to their appointment. 

## Details
The network takes in 9 input variables from 'data/trimmedApptData.csv' and creates
a number of hidden layers, as well as neurons per layer, passed in from the execution.
It uses backpropogation to calculate the error and propogate required changes to the 
weights in previous network layers in order to account for the difference in example and
network output. 

The network predicts whether or not a patient who scheduled an appointment is likely
to attend their scheduled appointment. The data is taken from the open source data 
repository kaggle:
https://www.kaggle.com/joniarroba/noshowappointments

9 attributes are calculated from the data:
```
1. sex			// male, female
2. days from sch-appt	// 0-5, 5-14, 14+
3. age			// unknown, under 25, 25-50, 50+
4. had scholarship	// yes, no
5. has hypertension	// yes, no
6. has diabetes		// yes, no
7. is alcholic		// yes, no
8. is handicapped	// yes, no
9. received reminder	// yes, no
```
## Important Notes
- The most thorough and complete version of the project is inside the batch_version folder.

## Use
Descend into the relevant directory of this repository to view different versions of 
the project. Inside each is a readme with instructions on running it. 

Custom Network Includes
```
- defined_versions: a simple, custom built and hardcoded size network
- ptr_versions:	allows for user-defined network layers and numbers
- vector_versions: allows for user defined network layers and numbers using vectors
- batch_version: parallel processing neural network trainer using openMP. represents
most complete work
```
## Sources
More specific sources are cited in comments.

## Authors
John Nemeth, Isaac Lance
