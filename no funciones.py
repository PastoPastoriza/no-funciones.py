def custom_loss(y_true, y_pred):
    # Define weights
    false_negative_weight = 1
    false_positive_weight = 3

    # Calculate binary cross entropy
    bce = tf.keras.losses.BinaryCrossentropy()

    # Calculate loss
    loss = bce(y_true, y_pred)

    # Calculate weighted loss
    weighted_loss = tf.where(tf.greater(y_true, y_pred), false_positive_weight * loss, false_negative_weight * loss)

    return tf.reduce_mean(weighted_loss)

import tensorflow as tf

def custom_loss2(y_true, y_pred):
    # Define weights
    false_negative_weight = 2
    false_positive_weight = 5

    # Calculate binary cross entropy
    bce = tf.keras.losses.BinaryCrossentropy()
    binary_crossentropy = bce(y_true, y_pred)

    # Calculate precision
    true_positives = tf.reduce_sum(tf.cast(y_true * tf.round(y_pred), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(tf.round(y_pred), tf.float32))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())

    # Calculate loss
    loss = binary_crossentropy + (1 - precision)

    # Calculate weighted loss
    weighted_loss = tf.where(tf.greater(y_true, y_pred), false_positive_weight * loss, false_negative_weight * loss)

    return tf.reduce_mean(weighted_loss)
------------------------------------

#si quiero bajar algunos 0s de array para el tensor de dsp

mul3 = 1
indices_to_delete=[]
for i in range(WINDOW-1,len(X_train),WINDOW):
  if y_train[i] == 0 and mul3 %3 != 0:
    indices_to_delete.extend(range(i-(WINDOW-1),i+1))
    mul3 +=1
  else:
    mul3 +=1
X_train = np.delete(X_train, indices_to_delete, axis=0)
y_train = np.delete(y_train, indices_to_delete, axis=0)

------------------------------------
#lo mismo de test
# mul3 = 1
# indices_to_delete=[]
# for i in range(WINDOW-1,len(X_test),WINDOW):
#   if y_test[i] == 0 and mul3 %3 != 0:
#     indices_to_delete.extend(range(i-(WINDOW-1),i+1))
#     mul3 +=1
#   else:
#     mul3 +=1
# X_test = np.delete(X_test, indices_to_delete, axis=0)
# y_test = np.delete(y_test, indices_to_delete, axis=0)


---------------------------------------

#si quiero bajar algunos 0s de array para ML
cero = np.where(y_train_a == 0)[0]
num_remove=int(0.4*len(cero))

i_remove = np.random.choice(cero,num_remove,replace=False)

X_train_a = np.delete(X_train_a,i_remove,axis=0)
y_train_a = np.delete(y_train_a,i_remove)
X_train_a.shape, y_train_a.shape


---------------------------------------
#Como contar 0 de tensores
cero_train = 0
one_train = 0

cero_test = 0
one_test = 0

for data, targets in train_data:
  for target in targets.numpy():
    if target == 0:
      cero_train += 1
    else:
      one_train +=1

for data, targets in test_data:
  for target in targets.numpy():
    if target == 0:
      cero_test += 1
    else:
      one_test +=1

cero_train, one_train, cero_test, one_test

---------------------------------------

import tensorflow as tf 

train_data = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=X_train_a,
    targets = y_train_a[WINDOW-1:],
    sequence_length = WINDOW,
    sequence_stride=WINDOW,
    sampling_rate=1,
    batch_size=BATCH_SIZE)

test_data = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=X_valid_a,
    targets = y_valid_a[WINDOW-1:],
    sequence_length = WINDOW,
    sequence_stride=WINDOW,
    sampling_rate=1,
    batch_size=BATCH_SIZE)

len(train_data), len(test_data)
------------------------------------------------------------------
# index_number = nas.data.index.get_loc(pd.to_datetime("2024-05-02 17:00"))
# print("Index number for the date {}: {}".format("2024-05-06 03:00", index_number))

------------------------------------------------------------------
# sns.reset_orig()
