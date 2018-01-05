import h5py
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Reshape, Dropout, GlobalAveragePooling2D, Merge, Concatenate, Input, concatenate, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.utils import plot_model
import os
import font_encoder

num_chars = 62
letter_width = 64
num_neurons = 4 * letter_width**2

data = h5py.File('fonts.hdf5')['fonts']

entry_dropout = 0.1
entry_pool_size = 4
entry_num_filters = 8

# http://bit.ly/2BI7srh
b_input = Input(shape=(64, 64, 1))
b_branch = Conv2D(entry_num_filters, (4,4), activation='relu')(b_input)
b_branch = MaxPooling2D(pool_size=entry_pool_size)(b_branch)
b_branch = Dropout(entry_dropout)(b_branch)

a_input = Input(shape=(64, 64, 1))
a_branch = Conv2D(entry_num_filters, (4,4), activation='relu')(a_input)
a_branch = MaxPooling2D(pool_size=entry_pool_size)(a_branch)
a_branch = Dropout(entry_dropout)(a_branch)

s_input = Input(shape=(64, 64, 1))
s_branch = Conv2D(entry_num_filters, (4,4), activation='relu')(s_input)
s_branch = MaxPooling2D(pool_size=entry_pool_size)(s_branch)
s_branch = Dropout(entry_dropout)(s_branch)

q_input = Input(shape=(64, 64, 1))
q_branch = Conv2D(entry_num_filters, (4,4), activation='relu')(q_input)
q_branch = MaxPooling2D(pool_size=entry_pool_size)(q_branch)
q_branch = Dropout(entry_dropout)(q_branch)

merged = concatenate([b_branch, a_branch, s_branch, q_branch])
merged = Conv2D(10, (4,4), activation='relu')(merged)
merged = MaxPooling2D(pool_size=(4, 4))(merged)
merged = GlobalAveragePooling2D()(merged)
merged = Dense(62*64*64, activation='sigmoid')(merged)
merged = BatchNormalization()(merged)
merged = Reshape((62, 64, 64))(merged)

model = Model(inputs=[b_input, a_input, s_input, q_input], output=merged)

# Save the architecture as a JSON file.
with open("model.json", "w") as f:
	f.write(model.to_json())

# Create the network.
model.compile(optimizer='sgd', metrics=['accuracy'], loss='mse')

b_inputs = []
a_inputs = []
s_inputs = []
q_inputs = []
outputs = []

for font in data:
	input_B = font[1].reshape(64, 64, 1)
	input_A = font[0].reshape(64, 64, 1)
	input_S = font[18].reshape(64, 64, 1)
	input_Q = font[16].reshape(64, 64, 1)

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

model.fit([b_inputs, a_inputs, s_inputs, q_inputs], outputs, epochs=70)
# Save the model.
model.save_weights("model.hdf5")

#print model.output_shape

# Plot the model so I can see what is going on.
if not os.path.exists("img"):
	os.makedirs("img")
plot_model(model, to_file='img/model.png')
plot_model(model, to_file='img/model.svg')

# Read and encode the test font.
test_font = font_encoder.read_font("UbuntuMono-R.ttf")
test_input = np.array([test_font[1].reshape(1, 64, 64, 1), test_font[0].reshape(1, 64, 64, 1), test_font[18].reshape(1, 64, 64, 1), test_font[16].reshape(1, 64, 64, 1)])
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