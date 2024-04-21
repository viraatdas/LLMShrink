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

def run_tflite_model(input_ids):
    pass

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

# Example usage
if __name__ == "__main__":
    # Input text
    input_text = "Hello, my name is "

    # Step 1: Encode the input
    input_ids = encode_input(input_text)

    # Step 2: Run the model
    output_logits = run_tf_model(input_ids)

    # Step 3: Decode the output
    output_text = decode_output(output_logits)

    print("Generated Text:\n", output_text)
