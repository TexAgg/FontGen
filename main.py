import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Reshape, Dropout, AveragePooling2D, Merge
from keras.utils import plot_model
import os
import font_encoder

num_chars = 62
letter_width = 64
num_neurons = 4 * letter_width**2

data = h5py.File('fonts.small.hdf5')['fonts']

b_branch = Sequential()
b_branch.add(Conv2D(2, (4,4), activation='relu', input_shape=(64,64,1), padding='same'))
#b_branch.add(AveragePooling2D(pool_size=(3, 3), padding='same'))
#b_branch.add(Dropout(0.25))

a_branch = Sequential()
a_branch.add(Conv2D(2, (4,4), activation='relu', input_shape=(64,64,1),padding='same'))
#a_branch.add(AveragePooling2D(pool_size=(3, 3), padding='same'))
#a_branch.add(Dropout(0.25))

s_branch = Sequential()
s_branch.add(Conv2D(2, (4,4), activation='relu', input_shape=(64,64,1), padding='same'))
#s_branch.add(AveragePooling2D(pool_size=(3, 3), padding='same'))
#s_branch.add(Dropout(0.25))

q_branch = Sequential()
q_branch.add(Conv2D(2, (4,4), activation='relu', input_shape=(64,64,1), padding='same'))
#q_branch.add(AveragePooling2D(pool_size=(3, 3), padding='same'))
#q_branch.add(Dropout(0.25))

# http://bit.ly/2lajrqn
merged = Merge([b_branch, a_branch, s_branch, q_branch], mode='concat')

# Create the network.
model = Sequential()
model.add(merged)
model.add(Conv2D(4,(3,3), activation='relu', padding='same'))
#model.add(AveragePooling2D(pool_size=(3, 3), padding='same'))
#model.add(Flatten())
model.add(Dense(62, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Reshape((62, 64, 64)))
model.compile(optimizer='sgd', metrics=['accuracy'], loss='mse')

b_inputs = []
a_inputs = []
s_inputs = []
q_inputs = []
outputs = []

for font in data:
	input_B = font[1].reshape(64,64,1)
	input_A = font[0].reshape(64,64,1)
	input_S = font[18].reshape(64,64,1)
	input_Q = font[16].reshape(64,64,1)

	# Now we need to flatten these fonts into one long numpy array.
	input_chars = np.array([input_B, input_A, input_S, input_Q])
	b_inputs.append(input_B)
	a_inputs.append(input_A)
	s_inputs.append(input_S)
	q_inputs.append(input_S)
	output_chars = font
	outputs.append(output_chars)

b_inputs = np.array(b_inputs)
a_inputs = np.array(a_inputs)
s_inputs = np.array(s_inputs)
q_inputs = np.array(q_inputs)
outputs = np.array(outputs)

model.fit([b_inputs, a_inputs, s_inputs, q_inputs], outputs, epochs=10)
# Save the model.
model.save_weights("model.hdf5")

print model.output_shape

# Plot the model so I can see what the hell is going on.
if not os.path.exists("img"):
	os.makedirs("img")
plot_model(model, to_file='img/model.png', show_shapes=True)
plot_model(model, to_file='img/model.svg', show_shapes=True)

# Read and encode the test font.
test_font = font_encoder.read_font("UbuntuMono-R.ttf")
test_input = np.array([test_font[1].reshape(1,64,64,1), test_font[0].reshape(1,64,64,1), test_font[18].reshape(1,64, 64, 1), test_font[16].reshape(1,64, 64, 1)])
#print test_input.shape

# Run the model on the test input.
test_output = model.predict([test_input[0], test_input[1], test_input[2], test_input[3]])
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