# HomeGymSpotterCNN
Working out at home but forgetting how many reps you've done? Worry no more.

-	During these Covid-19 times, the gyms are shut close. Therefore people have to rely on home workouts to keep fit. Most people have been doing pushups and squats, the most popular bodyweight exercises, at their homes. Personally, on some occasions, I found myself losing count of how many repetitions of pushups or squats I had done. I asked my friends about this and they said that they sometimes lose track of their repetitions too.
-	So I teamed up with a junior student and lead our team to design convolutional neural network that is aimed towards helping people keep count of how many reps of pushups/squats they have done.
-	This CNN is based on mobileNet v2 as it’s a lightweight network that can perform fast computations even on a phone.
-	The task was initially a classification problem in which the CNN has to classify if the person is gone up in their repetition, or down.
-	Using these predictions, a count is maintained and shown on the screen. With this count, the user can reliably track the repetitions of pushups or squats they have performed so far.
-	For the dataset, we recorded videos of us doing pushups and extracted each frame of the video and labelled them as up, down or neither. To make sure there was a variety in the training set, we recorded ourselves doing pushups in different lightings, background and clothes. This variety was needed to have a well generalised model.
-	To add more versatility and prevent overfitting, we also pre-processed the data with random rotations, random sheer, and random brightness changes.
-	The training set and test set were split in a 8:2 ratio. The total dataset size was 3200 images of pushups.
-	The pretrained weights of mobileNet v2 are publicly available that are trained on ImageNet Dataset, so we chose to perform transfer learning on our dataset instead of training it from scratch.
-	We simply used that pretrained model, froze all the weights and added 3 new layers with 20% dropout and relu activation. MobileNet v2 is designed to classify 1000 classes of ImageNet dataset but in this project the classes were only 3: up, down, neither. So we made sure the last layer had only 3 units instead of 1000.
-	We were getting better results with a regression based model instead of a classification based model so we changed the activation function of last layer to softmax and changed the loss to Categorical Cross Entropy loss function to better suit a regression model.
-	This transfer learned network had an accuracy of 92% on the test set.
