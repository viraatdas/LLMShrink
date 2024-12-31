import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

def encode_input(text):
    """
    Encodes the input text using GPT-2 tokenizer.
    
    Args:
    text (str): Input text to encode.

    Returns:
    tf.Tensor: Encoded text tensor suitable for GPT-2 input.
    """
    # Encode the text to get input IDs and return as TensorFlow tensors
    encoded_input = tokenizer(text, return_tensors='tf')
    return encoded_input['input_ids']

def run_tf_model(input_ids):
    """
    Runs the GPT-2 model on the encoded input IDs.
    
    Args:
    input_ids (tf.Tensor): Tensor of encoded input IDs.

    Returns:
    tf.Tensor: Output from the model (logits).
    """
    # Run the model and return the logits (outputs before softmax)
    output = model(input_ids)
    return output.logits


def convert_to_tflite(output_path='gpt2_model.tflite'):
    # Prepare the model for saving and conversion by setting up a serving function
    input_spec = tf.TensorSpec([1, None], dtype=tf.int32, name="input_ids")
    serving_fn = tf.function(lambda input_ids: model(input_ids)).get_concrete_function(input_spec)
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([serving_fn])
    tflite_model = converter.convert()
    
    # Save the converted model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print("Model has been converted to TFLite and saved.")

def decode_output(logits):
    """
    Decodes the output logits to text using the GPT-2 tokenizer.
    
    Args:
    logits (tf.Tensor): Logits tensor from the model output.

    Returns:
    str: Decoded text.
    """
    # Use TensorFlow's argmax to convert logits to token IDs and decode
    token_ids = tf.math.argmax(logits, axis=-1)
    decoded_text = tokenizer.decode(token_ids[0])
    return decoded_text


def load_tflite_model(model_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_tflite_model(interpreter, input_ids):
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_ids)

    # Run the model
    interpreter.invoke()

    # Retrieve the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Example usage with TFLite
if __name__ == "__main__":
    input_text = "Hello, my name is ChatGPT."
    input_ids = encode_input(input_text)  # Reuse the encode_input function from previous code

    # Run the TensorFlow model
    tf_output_logits = run_tf_model(input_ids)
    print("Generated TF output:\n", decode_output(tf_output_logits))


    # Load TFLite model
    tflite_model_path = 'gpt2_model.tflite'
    convert_to_tflite(output_path=tflite_model_path)

    tflite_interpreter = load_tflite_model(tflite_model_path)

    # Run the TFLite model
    output_logits = run_tflite_model(tflite_interpreter, input_ids.numpy())  # Make sure to pass a numpy array

    # Decode the output (reuse the decode_output function)
    output_text = decode_output(output_logits)
    print("Generated Text:", output_text)
