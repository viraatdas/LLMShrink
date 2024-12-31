import tensorflow as tf
import keras_nlp
import numpy as np

# Load components from KerasNLP
tokenizer = keras_nlp.tokenizers.GPT2Tokenizer.from_preset("gpt2_base_en")
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en", sequence_length=256, add_end_token=True)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en", preprocessor=preprocessor)


# Define a TensorFlow function that will be converted to TFLite
@tf.function
def generate(prompt):
    inputs = tokenizer([prompt])
    return gpt2_lm(inputs, training=False)

# Convert this model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_concrete_functions([generate.get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.string))])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True

# Convert the model
tflite_model = converter.convert()

# Save the model
with open("gpt2.tflite", "wb") as f:
    f.write(tflite_model)


# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the model input
input_data = np.array(["Hello, how are you?"], dtype=object)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run the model
interpreter.invoke()

# Get the model output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output:", output_data)

