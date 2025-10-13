import argparse
import torch
import numpy as np
from ProtSeqGen_code.Model.datasets import ts50_dataset 
from ProtSeqGen_code.Model.model import ProtSeqGen


def evaluate_test_set(args):
    """Evaluate ProtseqGen model on the TS50 test set and report loss, accuracy, and perplexity."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load TS50 dataset
    print("Loading TS50 dataset...")
    test_set = ts50_dataset(batch_size=args.batch_size)  
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # Load pretrained model checkpoint
    print("Loading model checkpoint...")
    checkpoint = torch.load(args.previous_checkpoint, map_location=device)

    # Initialize model
    model = ProtseqGen(
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_encoder_layers,
        k_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
    )

    # Load saved weights and move to device
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Initialize accumulators
    total_loss = 0
    total_acc = 0
    total_weights = 0
    
    # Evaluation loop
    with torch.no_grad():
        for batch in test_loader:
            X, S, mask = batch
            # Remove batch dimension and move to device
            X = X.squeeze(0).to(device)
            S = S.squeeze(0).long().to(device)
            mask = mask.squeeze(0).to(device)
            chain_M = torch.ones_like(S).to(device)
            mask_for_loss = mask * chain_M

            # Forward pass
            log_probs = model(X, S, mask, chain_M)
            # Compute negative log-likelihood loss and accuracy
            loss, _, true_false = loss_nll(S, log_probs, mask_for_loss)

            # Accumulate metrics
            total_loss += torch.sum(loss * mask_for_loss).item()
            total_acc += torch.sum(true_false * mask_for_loss).item()
            total_weights += torch.sum(mask_for_loss).item()
            
    # Compute average metrics
    test_loss = total_loss / total_weights
    test_accuracy = total_acc / total_weights
    test_perplexity = np.exp(test_loss)

    print(f"[TS50 EVALUATION]")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Perplexity: {test_perplexity:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model on the TS50 test set.")
    parser.add_argument("--path_for_outputs", type=str, required=True)
    parser.add_argument("--previous_checkpoint", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--num_neighbors", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--backbone_noise", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1000)

    args = parser.parse_args()
    evaluate_test_set(args)
