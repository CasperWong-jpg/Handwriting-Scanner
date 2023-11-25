# Handwriting-Scanner
Convert handwritten text to real text using custom-built neural network \
See [portfolio](https://casperwong.weebly.com/programming.html) for more details

### How to run
1. Extract MNIST36 data: `cd scripts/; source get_data.sh; cd ..`
2. Train neural network: `cd python; python3 train_nn.py` \
To benchmark performance of custom NN with NN built with Pytorch primitives, run: `python3 ./python/pytorch_benchmark.py`
3. Upload images to scan in `./images`
4. Run scanner: `python3 run_scanner.py`
