#!/usr/bin/env python3
"""
Script: finetune_dnabert_cfr_regressor_full_genome.py

Fine-tunes DNABERT on real + synthetic SARS-CoV-2 full-genome sequences to predict
global case-fatality rate (CFR) as a continuous value. All transformer layers are
unfrozen and a richer regression head (256→64→1) is used.

Usage:
    python finetune_dnabert_cfr_regressor_full_genome.py

Dependencies:
    tensorflow, transformers, biopython, numpy, scikit-learn, matplotlib
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
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    LearningRateScheduler
)
import matplotlib.pyplot as plt

# ─── Configuration ─────────────────────────────────────────────────────────────
ALIGNMENT_FASTA = 'aligned.fasta'
PRETRAINED       = 'zhihan1996/DNA_bert_6'
MAX_LEN          = 512           # maximum input length for DNABERT
BATCH_SIZE       = 2
EPOCHS           = 20
LR               = 2e-5
VALID_SPLIT      = 0.1
SEED             = 42
AUG_FACTOR       = 4
MUT_RATE         = 1e-4
OUTPUT_DIR       = 'dnabert_cfr_fullgenome_richer_head'
os.makedirs(OUTPUT_DIR, exist_ok=True)
tf_logging.set_verbosity_error()

# ─── Reproducibility ────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

# ─── Load and Map Alignment ─────────────────────────────────────────────────────
alignment = AlignIO.read(ALIGNMENT_FASTA, 'fasta')
ref = next(r for r in alignment if 'NC_045512.2' in r.id)

# Build a mapping from alignment column to reference genome position
mapping = {}
pos = 0
for i, nt in enumerate(ref.seq):
    if nt != '-':
        pos += 1
        mapping[i] = pos
    else:
        mapping[i] = None

# Keep every column that maps to a real base (entire genome)
offsets = [i for i, p in mapping.items() if p is not None]

# ─── Assemble Sequences & Labels ────────────────────────────────────────────────
global_cfr = {
    "WildType": 0.036, "Alpha": 0.026, "Beta": 0.042,
    "Gamma":   0.036, "Delta": 0.020, "Omicron": 0.007
}

def extract_variant(header):
    m = re.search(r"\[([^]]+)\]", header)
    return m.group(1) if m else None

real_seqs, real_labels = [], []
for rec in alignment:
    if 'NC_045512.2' in rec.id:
        continue
    var = extract_variant(rec.description)
    cfr = global_cfr.get(var)
    if cfr is None:
        continue
    # build ungapped full-genome string
    seq = ''.join(rec.seq[i] for i in offsets)
    real_seqs.append(seq)
    real_labels.append(cfr)

# Synthetic augmentation via random mutation
BASES = ['A','C','G','T']
def mutate_sequence(seq, rate):
    arr = list(seq)
    for i in range(len(arr)):
        if random.random() < rate:
            arr[i] = random.choice([b for b in BASES if b != arr[i]])
    return ''.join(arr)

aug_seqs, aug_labels = [], []
for seq, cfr in zip(real_seqs, real_labels):
    for _ in range(AUG_FACTOR):
        aug_seqs.append(mutate_sequence(seq, MUT_RATE))
        aug_labels.append(cfr)

all_seqs = real_seqs + aug_seqs
all_cfr  = np.array(real_labels + aug_labels, dtype=float)

# ─── Tokenization to 6-mers ─────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained(PRETRAINED, do_lower_case=False)
ids, masks = [], []
for seq in all_seqs:
    kmers = [seq[i:i+6] for i in range(len(seq)-5)]
    enc = tokenizer(
        ' '.join(kmers),
        max_length=MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='np'
    )
    ids.append(enc['input_ids'][0])
    masks.append(enc['attention_mask'][0])

X_ids  = np.stack(ids)
X_mask = np.stack(masks)
indices = np.arange(len(all_cfr))

# ─── Train/Val/Test Split ───────────────────────────────────────────────────────
trainval, test = train_test_split(indices, test_size=0.2, random_state=SEED)
train_i, val_i = train_test_split(trainval, test_size=VALID_SPLIT, random_state=SEED)

X_train = {'input_ids': X_ids[train_i], 'attention_mask': X_mask[train_i]}
y_train = all_cfr[train_i]
X_val   = {'input_ids': X_ids[val_i],   'attention_mask': X_mask[val_i]}
y_val   = all_cfr[val_i]
X_test  = {'input_ids': X_ids[test],    'attention_mask': X_mask[test]}
y_test  = all_cfr[test]

# ─── Model Definition ───────────────────────────────────────────────────────────
class DNABERTRegressorFullGenome(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.bert      = TFBertModel.from_pretrained(PRETRAINED, from_pt=True)
        # unfreeze entire model
        self.bert.trainable = True
        self.dense1    = tf.keras.layers.Dense(256, activation='relu')
        self.dropout1  = tf.keras.layers.Dropout(0.2)
        self.dense2    = tf.keras.layers.Dense(64, activation='relu')
        self.dropout2  = tf.keras.layers.Dropout(0.1)
        self.output_ln = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=False):
        out = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            training=training
        )
        x = out.pooler_output
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_ln(x)

model = DNABERTRegressorFullGenome()

model.compile(
    optimizer=Adam(learning_rate=LR, clipnorm=1.0),
    loss='mean_squared_error',
    metrics=[
        tf.keras.metrics.MeanSquaredError(name='mse'),
        tf.keras.metrics.RootMeanSquaredError(name='rmse')
    ]
)
model.summary()

# ─── Callbacks & LR Schedule ───────────────────────────────────────────────────
def step_decay(epoch, lr):
    if epoch > 0 and epoch % 5 == 0:
        new_lr = lr * 0.5
        print(f"\nEpoch {epoch}: reducing lr to {new_lr:.2e}")
        return new_lr
    return lr

callbacks = [
    LearningRateScheduler(step_decay, verbose=1),
    EarlyStopping(  monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-7, verbose=1),
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'best_model.h5'),
        monitor='val_loss', save_best_only=True, verbose=1
    )
]

# ─── Train & Evaluate ───────────────────────────────────────────────────────────
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)

# Save final model & tokenizer
model.save(os.path.join(OUTPUT_DIR, 'dnabert_fullgenome_richer_head.keras'))
tokenizer.save_pretrained(OUTPUT_DIR)

# ─── Final Metrics & Scatter Plot ───────────────────────────────────────────────
res = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print(f"Test MSE:  {res[0]:.6f}")
print(f"Test RMSE: {res[2]:.6f}")

y_pred = model.predict(X_test, batch_size=BATCH_SIZE).flatten()
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.3)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, '--', color='gray', linewidth=1)
plt.xlabel("True CFR")
plt.ylabel("Predicted CFR")
plt.title("Predicted vs True CFR (Full Genome)")
plt.tight_layout()
plt.show()
