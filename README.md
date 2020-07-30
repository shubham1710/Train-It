# Train It
A website where you can train my model for up to 5 different classes and get live predictions using their webcam.

Made with Tensorflow js and it uses transfer learning using the MobileNet Model. The upper layers are frozen and we train our new data on the output of the MobileNet Model and then proceed to produce the output.

The website has been deployed using Flask Framework which handles the login and signup events and changes the content based on authentication status of the user. It also deploys all the HTML pages.

Go on this link to test the model: 
