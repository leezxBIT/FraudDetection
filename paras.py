import argparse

parser = argparse.ArgumentParser()

# === Seed and basic info ===
parser.add_argument('--device', type=int, default=7)
parser.add_argument('--seed', type=int, default=42)

# === Dataset and data loader ===
parser.add_argument('--dataset', type=str, default='yelpchi')
parser.add_argument('--sample', type=str, default='neighbor', help='sampling method')
parser.add_argument('--saint_num_steps', type=int, default=5, help='number of steps for graphsaint')
parser.add_argument('--test_num_parts', type=int, default=10, help='number of partitions for testing')
parser.add_argument('--homo', type = bool, default=True, help='Whether of not homo')

# === Training strategies ===
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--train_ratio', type=float, default=.4)
parser.add_argument('--test_ratio', type=float, default=.4)
parser.add_argument('--model', type=str, default='Ours')
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default= 0.0005)
parser.add_argument('--batch_size', type=int, default=64) # 1024 for YelpChi, 256 for Amazon
parser.add_argument('--LT', type=str, default='CE', choices=['CE', 'RW', 'LA', 'ALA'])

# === LINXK parameters ===
parser.add_argument('--num_layers', type = int, default = 2)

args = parser.parse_args()
print('arguments\t', args)
