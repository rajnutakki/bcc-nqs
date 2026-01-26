#Example of running an optimization, then computing expectation values of optimized NQS
#First run an optimization
uv run ../../../src/vmc/optimization/run.py --config ../optimization/config.yaml
#Then compute expectation value of the optimized NQS
uv run ../../../src/vmc/expectation_value/run.py --config config.yaml
#Clean up optimization files 
uv run ../cleaner.py $PWD