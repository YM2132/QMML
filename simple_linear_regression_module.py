# function for analytical simple LR

def analytic_approach_simple_LR(x, y):
	xbar = x.mean()
	ybar = y.mean()

	# beta hat
	sum_numerator = 0
	sum_denominator = 0

	for xi, yi in zip(x, y):
		sum_numerator += (xi - xbar) * (yi - ybar)
		sum_denominator += (xi - xbar) ** 2

	betahat = sum_numerator / sum_denominator

	# alpha hat
	alphahat = ybar - (betahat * xbar)

	return alphahat, betahat


def iterative_approach_simple_LR(x, y, iterations, lr):
	# The iterative approach using gradient descent

	alphahat = 0
	betahat = 0

	n = len(x)

	# iterations = 10000
	learning_rate = lr

	for i in range(iterations):
		y_preds = alphahat + betahat * x

		# Because we are using numpy arrays the y - y_preds will operate element wise we could use for loops to
		# do the same thing here
		d_alpha = -2 / n * sum(y - y_preds)
		d_beta = -2 / n * sum(x * (y - y_preds))

		# update alpha and beta
		alphahat -= learning_rate * d_alpha
		betahat -= learning_rate * d_beta

	return alphahat, betahat