"""
Step 2: train TinyPhaseNet on the 'train' split (earthquake + noise, random windows),
validate on the 'val' split (locked windows). The 'test' split is not touched here.
"""
import os
import tensorflow as tf
import config as C
from data import load_stead, eval_windows, TrainSequence
from model import create_tinyphasenet

if __name__ == '__main__':
    tf.keras.utils.set_random_seed(C.SEED)
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    meta, z = load_stead()
    train_seq = TrainSequence(meta[meta.split == 'train'], z)
    X_val, _, Y_val = eval_windows(meta[meta.split == 'val'], z, with_targets=True, prefilter_input=False)
    print(f"train: {len(train_seq) * C.BATCH_SIZE} windows/epoch, val: {len(X_val)} jendela")

    model = create_tinyphasenet()
    model.compile(optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE), loss='categorical_crossentropy')
    model.summary()
    print(f"Total parameter: {model.count_params():,}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(C.MODEL_DIR, 'tinyphasenet_best.keras'),
                                           monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=C.EARLY_STOP_PATIENCE,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(C.RESULTS_DIR, 'train_history.csv')),
    ]
    model.fit(train_seq, validation_data=(X_val, Y_val), batch_size=C.BATCH_SIZE,
              epochs=C.EPOCHS, callbacks=callbacks, verbose=2)
    print("SELESAI.")
