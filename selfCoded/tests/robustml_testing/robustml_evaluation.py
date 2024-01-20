# Create an instance of the wrapped model
#adjust dataset and threat_model

import robustml
robustml_model = RobustMLResNet18(model, dataset='CIFAR-10', threat_model='Linf')

# Define the threat model parameters (e.g., Linf norm, epsilon value)
threat_model = robustml.threat_model.Linf(epsilon=0.3)

# Create an evaluation instance with the model, dataset, and threat model
evaluation = robustml.evaluation.Evaluation(model=robustml_model, dataset=testset, threat_model=threat_model)

# Run the evaluation
evaluation.run()

# Optionally, you can access detailed results
results = evaluation.results()