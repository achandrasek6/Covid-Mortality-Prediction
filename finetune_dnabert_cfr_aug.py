#!/usr/bin/env python3
"""
finetune_dnabert_cfr_regressor_last4.py

Fine-tunes DNABERT on real + synthetic Spike sequences to predict
global case-fatality rate (CFR) as a continuous value, unfreezing
only the last 4 transformer layers for faster training. Uses:
- Reduced sequence length (256 tokens) and small batch (2) for memory
- Step-decay learning-rate schedule (halve every 5 epochs)
- Mean Squared Error loss for regression
- Gradient clipping and standard callbacks
"""

import os
import random
import re
import numpy as np
import tensorflow as tf
from Bio import AlignIO
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel, logging as tf_logging
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
)

# ---- Configuration ----
ALIGNMENT_FASTA = 'aligned.fasta'
PRETRAINED       = 'zhihan1996/DNA_bert_6'
MAX_LEN          = 256    # reduced for memory
BATCH_SIZE       = 2
EPOCHS           = 20
LR               = 2e-5
VALID_SPLIT      = 0.1
SEED             = 42
AUG_FACTOR       = 4
MUT_RATE         = 1e-4
OUTPUT_DIR       = 'dnabert_cfr_regressor_last4'
os.makedirs(OUTPUT_DIR, exist_ok=True)
tf_logging.set_verbosity_error()

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("GPUs available:", tf.config.list_physical_devices('GPU'))

# ---- Data Preparation ----
alignment = AlignIO.read(ALIGNMENT_FASTA, 'fasta')
START, END = 21562, 25384
global_cfr = {
    "WildType": 0.036, "Alpha": 0.026, "Beta": 0.042,
    "Gamma":   0.036, "Delta": 0.020, "Omicron": 0.007
}

def extract_variant(header):
    m = re.search(r"\[([^]]+)\]", header)
    return m.group(1) if m else None

# map coding positions
ref = next(r for r in alignment if 'NC_045512.2' in r.id)
mapping, count = {}, 0
for i, nt in enumerate(ref.seq):
    mapping[i] = (count := count + 1) if nt != '-' else None
offsets = [i for i,pos in mapping.items() if pos and START <= pos <= END]

# extract real sequences and numeric labels
real_seqs, real_labels = [], []
for rec in alignment:
    var = extract_variant(rec.description)
    if var in global_cfr and 'NC_045512.2' not in rec.id:
        seq = ''.join(rec.seq[i] for i in offsets).replace('-', '')
        real_seqs.append(seq)
        real_labels.append(global_cfr[var])

# synthetic augmentation
BASES = ['A','C','G','T']
def mutate_sequence(seq, rate):
    s = list(seq)
    for j in range(len(s)):
        if random.random() < rate:
            s[j] = random.choice([b for b in BASES if b != s[j]])
    return ''.join(s)

aug_seqs, aug_labels = [], []
for seq, cfr in zip(real_seqs, real_labels):
    for _ in range(AUG_FACTOR):
        aug_seqs.append(mutate_sequence(seq, MUT_RATE))
        aug_labels.append(cfr)

# combine sequences and labels
all_seqs = real_seqs + aug_seqs
all_cfr  = np.array(real_labels + aug_labels, dtype=float)

# ---- Tokenization ----
tokenizer = BertTokenizer.from_pretrained(PRETRAINED, do_lower_case=False)
ids, masks = [], []
for seq in all_seqs:
    kmers = [seq[i:i+6] for i in range(len(seq)-5)]
    enc = tokenizer(' '.join(kmers),
                    max_length=MAX_LEN,
                    truncation=True,
                    padding='max_length',
                    return_tensors='np')
    ids.append(enc.input_ids[0])
    masks.append(enc.attention_mask[0])

X_ids  = np.stack(ids)
X_mask = np.stack(masks)
idx    = np.arange(len(all_cfr))

# train/val/test split
trainval, test = train_test_split(idx, test_size=0.2, random_state=SEED)
train_i, val_i = train_test_split(trainval, test_size=VALID_SPLIT, random_state=SEED)

X_train = {'input_ids': X_ids[train_i], 'attention_mask': X_mask[train_i]}
y_train = all_cfr[train_i]
X_val   = {'input_ids': X_ids[val_i],   'attention_mask': X_mask[val_i]}
y_val   = all_cfr[val_i]
X_test  = {'input_ids': X_ids[test],    'attention_mask': X_mask[test]}
y_test  = all_cfr[test]

# ---- Model Definition ----
class DNABERTRegressor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.bert = TFBertModel.from_pretrained(PRETRAINED, from_pt=True)
        self.bert.trainable = False
        # unfreeze last 4 transformer layers
        for layer in self.bert.bert.encoder.layer[-4:]:
            layer.trainable = True
        self.dense   = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.out     = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=False):
        out    = self.bert(input_ids=inputs['input_ids'],
                           attention_mask=inputs['attention_mask'],
                           training=training)
        pooled = out.pooler_output
        x      = self.dense(pooled)
        x      = self.dropout(x, training=training)
        return self.out(x)

model = DNABERTRegressor()

# ---- Compile ----
optimizer = Adam(learning_rate=LR, clipnorm=1.0)
model.compile(
    optimizer=optimizer,
    loss='mean_squared_error',
    metrics=[
        tf.keras.metrics.MeanSquaredError(name='mse'),
        tf.keras.metrics.RootMeanSquaredError(name='rmse')
    ]
)

# ---- Learning-rate schedule ----
def step_decay(epoch, lr):
    if epoch > 0 and epoch % 5 == 0:
        new_lr = lr * 0.5
        print(f"\nEpoch {epoch}: reducing lr to {new_lr:.2e}")
        return new_lr
    return lr

lr_callback = LearningRateScheduler(step_decay, verbose=1)

# ---- Callbacks ----
cbs = [
    lr_callback,
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-7, verbose=1),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_model.h5'),
                    monitor='val_loss', save_best_only=True, verbose=1)
]

# ---- Train & Evaluate ----
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cbs,
    verbose=2
)

# ---- Save ----
model.save(os.path.join(OUTPUT_DIR, 'dnabert_cfr_regressor_last4.keras'))
tokenizer.save_pretrained(OUTPUT_DIR)

# ---- Final eval ----
res = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print(f"Test MSE: {res[0]:.6f}, Test RMSE: {res[1]:.6f}")
