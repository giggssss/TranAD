import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='Gyeongsan',
                    help="dataset from ['Gyeongsan', 'SWaT', 'WADI', 'SMD']")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='TranAD',
                    help="model name")
parser.add_argument('--epoch', 
					metavar='-e', 
					type=int, 
					required=False,
					default='5',
                    help="train epoch")

parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
# parser.set_defaults(test=True)

parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
args = parser.parse_args()