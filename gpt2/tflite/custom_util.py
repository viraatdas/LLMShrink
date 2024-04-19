import tensorflow as tf

def custom_cross_entropy(y_true, y_pred, ignore_index=-1):
    # Create a mask by comparing the target tensor with the ignore_index
    mask = tf.not_equal(y_true, ignore_index)
    
    # Convert mask to 1s and 0s (True becomes 1, False becomes 0)
    mask = tf.cast(mask, dtype=tf.float32)
    
    # Flatten the mask and the true labels
    y_true_flattened = tf.reshape(y_true, [-1])
    mask_flattened = tf.reshape(mask, [-1])
    
    # Calculate cross-entropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_flattened, logits=y_pred)
    
    # Apply the mask to the loss
    loss *= mask_flattened
    
    # Average the loss, but only over non-ignored entries
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask_flattened)
    
    return loss