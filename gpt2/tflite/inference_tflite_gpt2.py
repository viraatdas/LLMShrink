from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

model_path = 'gpt2_model.tflite'

def convert_model_to_tflite(output_path='gpt2_model.tflite'):
    # Load the model and tokenizer
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Encode input context
    text = "Once upon a time in a land far, far away,"  # You can replace this with any text seed
    encoded_input = tokenizer.encode(text, return_tensors='tf')

    # Generate text using the model
    output_sequences = model.generate(
        input_ids=encoded_input,
        max_length=100,  # Specifies the maximum length of the output
        num_return_sequences=1,  # Number of sentences to generate
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        top_p=0.92,
        temperature=0.85,
        do_sample=True,
        top_k=50
    )

    # Decode the output to strings
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    print(generated_text)

    # Assume a typical input shape, e.g., batch size of 1 and sequence length of 128
    input_spec = tf.TensorSpec([1, 128], tf.int32)
    model._set_inputs(input_spec)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model

# Load the TFLite model from the file
def inference_tflite(model_path, text):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Encode the text using the same tokenizer used during training
    encoded_input = tokenizer.encode(text, return_tensors='tf')
    input_data = encoded_input.numpy()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    # interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # Retrieve the model output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Assuming output_data contains token IDs
    print("output data is ")
    # print(output_data)
    # predicted_text = tokenizer.decode(output_data[0])
    # return predicted_text
    return output_data


convert_model_to_tflite(model_path)

text = "Hello, my name is "
output = inference_tflite(model_path, text)
print(output)







