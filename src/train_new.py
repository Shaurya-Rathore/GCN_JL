from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from metric import accuracy, roc_auc_compute_fn
from models import *
from earlystopping import EarlyStopping
from sample import Sampler
from utils import jl_project_features
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ------------------------------
# Unified Projection Function
# ------------------------------
def project_features(X_tensor, args):
    """
    Applies dimensionality reduction to features.
    Supports: none, JL, PCA, t-SNE.
    X_tensor: torch.FloatTensor [N, d_in]
    Returns: torch.FloatTensor [N, d_out]
    """
    method = args.proj.lower()
    d_in = X_tensor.size(1)

    # 1) JL projection
    if method == "jl" and args.proj_dim > 0 and d_in > args.proj_dim:
        Z, _ = jl_project_features(
            X_tensor,
            d_out=args.proj_dim,
            seed=args.jl_seed,
            orthogonalize=args.jl_orth
        )
        print(f"[JL] Projected {d_in} -> {args.proj_dim}")
        return Z

    # 2) PCA
    elif method == "pca" and args.proj_dim > 0 and d_in > args.proj_dim:
        print(f"[PCA] Computing PCA to {args.proj_dim} dims...")
        X_np = X_tensor.cpu().numpy()
        pca = PCA(n_components=args.proj_dim, whiten=args.pca_whiten, random_state=args.jl_seed)
        Z = pca.fit_transform(X_np)
        return torch.tensor(Z, dtype=torch.float32, device=X_tensor.device)

    # 3) t-SNE
    elif method == "tsne":
        print(f"[t-SNE] Computing t-SNE to {args.proj_dim} dims...")
        X_np = X_tensor.cpu().numpy()
        tsne = TSNE(
            n_components=args.proj_dim,
            perplexity=args.tsne_perplexity,
            learning_rate=args.tsne_lr,
            random_state=args.jl_seed,
            init="random"
        )
        Z = tsne.fit_transform(X_np)
        return torch.tensor(Z, dtype=torch.float32, device=X_tensor.device)

    # 4) No projection
    else:
        print(f"[No Projection] Keeping features as {d_in}D.")
        return X_tensor


# ------------------------------
# Argument Parser
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--fastmode', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lradjust', action='store_true', default=False)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument("--mixmode", action="store_true", default=False)
parser.add_argument("--warm_start", default="")
parser.add_argument('--debug', action='store_true', default=True)
parser.add_argument('--dataset', default="cora")
parser.add_argument('--datapath', default="src/data/")
parser.add_argument("--early_stopping", type=int, default=50)
parser.add_argument('--type', default="resgcn")
parser.add_argument('--inputlayer', default='gcn')
parser.add_argument('--outputlayer', default='gcn')
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--withbias', action='store_true', default=False)
parser.add_argument('--withbn', action='store_true', default=False)
parser.add_argument('--withloop', action="store_true", default=False)
parser.add_argument('--nhiddenlayer', type=int, default=1)
parser.add_argument("--normalization", default="AugNormAdj")
parser.add_argument("--nbaseblocklayer", type=int, default=16)
parser.add_argument("--aggrmethod", default="default")
parser.add_argument("--b", type=float, default=0.1)
parser.add_argument("--a", type=float, default=1.0)
parser.add_argument("--alpha", type=float, default=0.2)

# Projection arguments
parser.add_argument('--proj', type=str, default='jl',
                    choices=['none', 'jl', 'pca', 'tsne'],
                    help='Projection method.')
parser.add_argument('--proj-dim', type=int, default=512)
parser.add_argument('--jl-seed', type=int, default=0)
parser.add_argument('--jl-orth', action='store_true', default=True)
parser.add_argument('--pca-whiten', action='store_true', default=False)
parser.add_argument('--tsne-perplexity', type=float, default=30.0)
parser.add_argument('--tsne-lr', type=float, default=200.0)

args = parser.parse_args()
print(args)


# ------------------------------
# Setup
# ------------------------------
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.aggrmethod == "default":
    args.aggrmethod = "add" if args.type == "resgcn" else "concat"

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

sampler = Sampler(args.dataset, args.datapath)

adj = sampler.train_adj.tocoo()
data = [np.dot(sampler.ori_features[adj.row[i]], sampler.ori_features[adj.col[i]]) + 1e-10
        for i in range(len(adj.row))]
sampler.mask = sp.coo_matrix((data, (adj.row, adj.col)), shape=adj.shape)
sample_method = sampler.get_sample_func()


def main():
    print("CUDA available?", torch.cuda.is_available())
    labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(args.cuda)
    nfeat = sampler.nfeat
    nclass = sampler.nclass

    # Define model input dimension
    effective_nfeat = min(args.proj_dim, nfeat) if args.proj_dim > 0 else nfeat

    model = GCNModel(
        nfeat=effective_nfeat,
        nhid=args.hidden,
        nclass=nclass,
        nhidlayer=args.nhiddenlayer,
        dropout=args.dropout,
        baseblock=args.type,
        inputlayer=args.inputlayer,
        outputlayer=args.outputlayer,
        nbaselayer=args.nbaseblocklayer,
        activation=F.relu,
        withbn=args.withbn,
        withloop=args.withloop,
        withbias=args.withbias,
        aggrmethod=args.aggrmethod,
        mixmode=args.mixmode
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 400], gamma=0.5)

    if args.cuda:
        model.cuda()
        labels, idx_train, idx_val, idx_test = labels.cuda(), idx_train.cuda(), idx_val.cuda(), idx_test.cuda()

    if args.early_stopping > 0:
        early_stopping = EarlyStopping(patience=args.early_stopping, verbose=False)
        print(f"Model checkpoint: {early_stopping.fname}")

    def get_lr(opt):
        return opt.param_groups[0]['lr']

    def train(train_adj, train_fea, idx_train, val_adj, val_fea):
        model.train()
        optimizer.zero_grad()
        output = model(train_fea, train_adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        output_val = model(val_fea, val_adj)
        loss_val = F.nll_loss(output_val[idx_val], labels[idx_val]).item()
        acc_val = accuracy(output_val[idx_val], labels[idx_val]).item()

        return loss_train.item(), acc_train.item(), loss_val, acc_val, get_lr(optimizer)

    def test(test_adj, test_fea):
        model.eval()
        output = model(test_fea, test_adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        auc_test = roc_auc_compute_fn(output[idx_test], labels[idx_test])
        print(f"Test Results: loss={loss_test:.4f}, auc={auc_test:.4f}, acc={acc_test:.4f}")

    # Train loop
    for epoch in range(args.epochs):
        (train_adj, train_fea) = sample_method(
            normalization=args.normalization,
            cuda=args.cuda,
            layer_num=args.nbaseblocklayer + 2,
            epoch=epoch,
            b=args.b, a=args.a
        )

        train_fea = project_features(train_fea, args)
        val_adj, val_fea = sampler.get_test_set(args.normalization, args.cuda)
        val_fea = project_features(val_fea, args)

        loss_t, acc_t, loss_v, acc_v, lr = train(train_adj, train_fea, idx_train, val_adj, val_fea)
        print(f"Epoch {epoch:03d}: train_loss={loss_t:.4f}, val_loss={loss_v:.4f}, val_acc={acc_v:.4f}, lr={lr:.5f}")

    print("Optimization Finished!")
    test_adj, test_fea = sampler.get_test_set(args.normalization, args.cuda)
    test_fea = project_features(test_fea, args)
    test(test_adj, test_fea)


if __name__ == "__main__":
    main()
