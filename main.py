import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Reshape, Dropout
from keras.utils import plot_model
import os
import font_encoder

num_chars = 62
letter_width = 64
num_neurons = 4 * letter_width**2

data = h5py.File('fonts.hdf5')['fonts']

# Create the network.
model = Sequential()
model.add(Conv2D(4,(4,4), activation='relu', input_shape=(4,64,64)))
model.add(Dropout(0.75))
model.add(Flatten())
model.add(Dense(64*64*62, activation='sigmoid'))
model.add(Dropout(0.85))
model.add(Reshape((62, 64, 64)))
model.compile(optimizer='sgd', metrics=['accuracy'], loss='mse')

inputs = []
outputs = []

for font in data:
	input_b = font[1]
	input_a = font[0]
	input_s = font[18]
	input_q = font[16]

	# Now we need to flatten these fonts into one long numpy array.
	input_chars = np.array([input_b, input_a, input_s, input_q])
	output_chars = font
	#print len(input_chars), len(output_chars)
	inputs.append(input_chars)
	outputs.append(output_chars)
	#model.train_on_batch(input_chars, output_chars)

inputs = np.array(inputs)
outputs = np.array(outputs)
#print inputs.shape, outputs.shape

model.fit(inputs, outputs, epochs=10)
# Save the model.
model.save_weights("model.hdf5")

#print model.output_shape

# Plot the model so I can see what the hell is going on.
if not os.path.exists("img"):
	os.makedirs("img")
plot_model(model, to_file='img/model.png', show_shapes=True)
plot_model(model, to_file='img/model.svg', show_shapes=True)

# Read and encode the test font.
test_font = font_encoder.read_font("UbuntuMono-R.ttf")
test_input = np.array([test_font[1], test_font[0], test_font[18], test_font[16]]).reshape((1,4,64,64))
#print test.shape

# Run the model on the test input.
test_output = model.predict(test_input)
# Save the output and input so they can be analyzed.
f = h5py.File("test.hdf5", "w")
dset = f.create_dataset("output", data=test_output)
f.create_dataset("input", data=test_input)

if not os.path.exists("out"):
	os.makedirs("out")
for font in test_output:
	for j in range(font.shape[0]):
		fname = 'out/{}.jpg'.format(font_encoder.chars[j])
		font_encoder.save_font(font[j], fname)