import pickle

import dynesty
from dynesty import plotting as dyplot

import matplotlib.pyplot as plt

import numpy as np

class ContinuousResults(object):

	def __init__(self, results_pkl):

		self.results_pkl = results_pkl

		pickle_in = open("{}".format(self.results_pkl),"rb")
		self.results = pickle.load(pickle_in)

		self.weights = np.exp(self.results['logwt'] - self.results['logz'][-1])

		self.get_best_fitting_parameters()


	def show_corner(self, savefile=None):

		cfig, caxes = dyplot.cornerplot(self.results, color='black', labels=[r'Av', r'logz', 
										r'B', r'detla'], 
										label_kwargs={'fontsize':25},
										show_titles=True)

		if savefile:
			cfig.savefig(savefile)
		else:
			plt.show()


	def get_best_fitting_parameters(self):
		
		self.av = dynesty.utils.quantile(x=self.results['samples'][:,0], 
					q=[0.16,0.50,0.84], weights=self.weights)

		self.logz = dynesty.utils.quantile(x=self.results['samples'][:,1], 
			q=[0.16,0.50,0.84], weights=self.weights)

		self.bump = dynesty.utils.quantile(x=self.results['samples'][:,2], 
			q=[0.16,0.50,0.84], weights=self.weights)

		self.delta = dynesty.utils.quantile(x=self.results['samples'][:,3], 
			q=[0.16,0.50,0.84], weights=self.weights)


	def get_logz_distribution(self):
		
		samples_equal = dynesty.utils.resample_equal(self.results['samples'], self.weights)

		self.logz_dist = np.empty(samples_equal.shape[0])

		for i in range(samples_equal.shape[0]):
			self.logz_dist[i] = samples_equal[i,1]


class BurstWeightsResults(object):

	def __init__(self, results_pkl):

		self.results_pkl = results_pkl

		pickle_in = open("{}".format(self.results_pkl),"rb")
		self.results = pickle.load(pickle_in)

		self.weights = np.exp(self.results['logwt'] - self.results['logz'][-1])


	def show_corner(self, savefile=None):

		cfig, caxes = dyplot.cornerplot(self.results)

		if savefile:
			cfig.savefig(savefile)
		else:
			plt.show()


	def show_runplot(self, savefile=None):

		fig, axes = dyplot.runplot(self.results)
		plt.show()


	def get_best_fit_metallicity(self, setllar_models, ndim):

		samples_equal = dynesty.utils.resample_equal(self.results['samples'], self.weights)

		z_final = np.empty(samples_equal.shape[0])

		for i in range(samples_equal.shape[0]):

			xi = np.empty(ndim)
			zi = np.empty(ndim)

			for j, (k, v) in enumerate(setllar_models.items()):

				xi[j] = 10**samples_equal[:,j][i]
				zi[j] = v['z']
				#print(xi[j], zi[j])

			z_final[i] = np.sum(xi * zi) / np.sum(xi)


		self.z_dist = z_final
		self.z = np.mean(z_final)
		self.z_err = np.std(z_final)
		self.logz = np.log10(self.z)
		self.logz_err = self.z_err / self.z


	def get_best_fit_age(self, setllar_models, ndim):

		samples_equal = dynesty.utils.resample_equal(self.results['samples'], self.weights)

		age_final = np.empty(samples_equal.shape[0])

		for i in range(samples_equal.shape[0]):

			xi = np.empty(ndim)
			agei = np.empty(ndim)

			for j, (k, v) in enumerate(setllar_models.items()):

				xi[j] = 10**samples_equal[:,j][i]
				agei[j] = v['age']

			age_final[i] = np.sum(xi * agei) / np.sum(xi)


		self.age_dist = age_final
		self.age = np.mean(age_final)
		self.age_err = np.std(age_final)







