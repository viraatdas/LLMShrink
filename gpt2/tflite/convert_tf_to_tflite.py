from transformers import TFGPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import tensorflow as tf

#### Load the TF model 


# Load the TensorFlow model
config = GPT2Config.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tf_model = TFGPT2LMHeadModel.from_pretrained('gpt2', config=config)

# Input text
input_text = "The meaning of life is "

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='tf')

# Generate text
output = tf_model.generate(
    input_ids,
    max_length=50,  # Maximum length of the generated text
    do_sample=True,  # Enable sampling for better results
    top_k=50,  # Top-k sampling
    top_p=0.95,  # Nucleus sampling
    num_return_sequences=1  # Number of generated sequences
)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)



#### Convert the model to a TF Lite model

# Set the model to inference mode
tf_model.set_weights(tf_model.get_weights())

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Use float16 for better performance
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('gpt_2.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved to gpt2_tflite_model.tflite")