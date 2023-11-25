# Handwriting-Scanner
Software that is able to convert handwritten text to real text using custom-built neural network
See [portfolio](https://casperwong.weebly.com/programming.html) for more details

### How to run
1. Extract MNIST36 data: `source ./scripts/get_data.sh`
2. Train neural network: `python3 ./python/train_nn` \
To benchmark performance of custom NN with NN built with Pytorch primitives, run: `python3 ./python/pytorch_benchmark.py`
3. Upload images to scan in `./images`
4. Run scanner: `python3 ./python/run_scanner`