import torch
import torch.nn as nn
import pickle
import os
from django.shortcuts import render

# Charger le modèle et les vocabulaires
device = torch.device("cpu")
model_dir = "model"

# Charger les vocabulaires
with open(os.path.join(model_dir, "vocab_darija.pkl"), "rb") as f:
    vocab_darija = pickle.load(f)
with open(os.path.join(model_dir, "vocab_eng.pkl"), "rb") as f:
    vocab_eng = pickle.load(f)

# Inverser vocab_eng pour obtenir un mapping index -> mot pour la traduction
idx_to_word_eng = {index: word for word, index in vocab_eng.items()}

# Définition des classes du modèle
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, trg, hidden, cell):
        trg = trg.unsqueeze(1)
        embedded = self.embedding(trg)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        trg_token = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(trg_token, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1)
            trg_token = trg[:, t] if np.random.random() < teacher_forcing_ratio else top1
        return outputs

# Fonction pour encoder une phrase
def encode_sentence(sentence, vocab, max_len):
    tokens = sentence.split()
    encoded = [vocab.get(word, vocab["<unk>"]) for word in tokens]
    # Ajouter des tokens <bos> et <eos> et appliquer le padding
    encoded = [vocab["<bos>"]] + encoded + [vocab["<eos>"]]
    encoded += [vocab["<pad>"]] * (max_len - len(encoded))
    return encoded[:max_len]

# Fonction pour traduire une phrase
def translate_sentence(sentence, model, input_vocab, output_vocab, idx_to_word, max_len=20):
    tokens = encode_sentence(sentence, input_vocab, max_len)
    src = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src)
        trg_indices = [output_vocab["<bos>"]]

        for _ in range(max_len):
            trg_tensor = torch.tensor([trg_indices[-1]]).to(device)
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            pred_token = output.argmax(1).item()
            trg_indices.append(pred_token)
            if pred_token == output_vocab["<eos>"]:
                break

    translated_sentence = [idx_to_word[idx] for idx in trg_indices[1:-1]]  # Retirer <bos> et <eos>
    return " ".join(translated_sentence)

def traducteur_darija(request):
    if request.method == "POST":
        darija_text = request.POST.get("darija_text")

        # Initialiser le modèle
        input_dim = len(vocab_darija)
        output_dim = len(vocab_eng)
        emb_dim = 256
        hidden_dim = 512

        encoder = Encoder(input_dim, emb_dim, hidden_dim)
        decoder = Decoder(output_dim, emb_dim, hidden_dim)
        model = Seq2Seq(encoder, decoder, device).to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, "seq2seq_model.pth"), map_location=device))
        model.eval()  # Mode évaluation pour la prédiction

        translation = translate_sentence(darija_text, model, vocab_darija, vocab_eng, idx_to_word_eng)
        return render(request, "traduction/home.html", {"translation": translation})
    return render(request, "traduction/home.html")