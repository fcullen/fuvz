import numpy as np
import dynesty

from . import utils
from . import model as fuvmod

import matplotlib.pyplot as plt

import pickle

class Continous(object):
	"""
	A fit using consitious SFH models as in Cullen et al. 2019
	"""

	def __init__(self, model, model_interpolation, fuv_mask_type='full', stellar_only=False):

		self.model = model
		self.model_interpolation = model_interpolation

		self.fuv_mask_type = fuv_mask_type

		self.ndim = 4

		self.stellar_only = stellar_only

		# intialize the uv mask to None
		self.fuv_mask = None


	def input_observations(self, wl, flam, flam_err):

		self.wl_obs = wl
		self.flam_obs = flam
		self.flam_obs_err = flam_err


	def get_fuv_continuum_mask(self, wl):
		'''
		For an input wavelegnth grid (wl), returns a mask of
		the purely stellar UV windows from Steidel + 2016
		'''
			
		m1 = (wl > 1042.) & (wl < 1080.)
		m2 = (wl > 1086.) & (wl < 1119.)
		m3 = (wl > 1123.) & (wl < 1131.)
		m4 = (wl > 1134.) & (wl < 1141.)
		m5 = (wl > 1144.) & (wl < 1186.)
		m6 = (wl > 1198.) & (wl < 1202.)
		m7 = (wl > 1221.) & (wl < 1254.)
		m8 = (wl > 1270.) & (wl < 1291.)
		m9 = (wl > 1312.) & (wl < 1328.)
		m10 = (wl > 1340.) & (wl < 1363.)
		m11 = (wl > 1373.) & (wl < 1389.)
		m12 = (wl > 1396.) & (wl < 1398.)
		m13 = (wl > 1404.) & (wl < 1521.)
		m14 = (wl > 1528.) & (wl < 1531.)
		m15 = (wl > 1536.) & (wl < 1541.)
		m16 = (wl > 1552.) & (wl < 1606.)
		m17 = (wl > 1610.) & (wl < 1657.)
		m18 = (wl > 1675.) & (wl < 1708.)
		m19 = (wl > 1711.) & (wl < 1740.)
		m20 = (wl > 1743.) & (wl < 1751.)
		m21 = (wl > 1754.) & (wl < 1845.)
		m22 = (wl > 1864.) & (wl < 1878.)
		m23 = (wl > 1885.) & (wl < 1903.)
		m24 = (wl > 1920.) & (wl < 2000.)
		m25 = (wl > 2000.) & (wl < 2300.)

		if self.fuv_mask_type == 'full': 
			mask = m7 | m8 | m9 | m10 | m11 | m12 | m13 | m14 | m15 | \
				m16 | m17 | m18 | m19 | m20 | m21 | m22 | m23 | m24
		elif self.fuv_mask_type == 'exclude_heii':
			mask = m7 | m8 | m9 | m10 | m11 | m12 | m13 | m14 | m15 | \
				m16 | m18 | m19 | m20 | m21 | m22 | m23 | m24
		elif self.fuv_mask_type == 'exclude_nv_heii':
			mask = m8 | m9 | m10 | m11 | m12 | m13 | m14 | m15 | \
				m16 | m18 | m19 | m20 | m21 | m22 | m23 | m24

		return mask


	def normalize_model_to_spec(self, fmodel):

		num = np.sum((fmodel * self.flam_obs[self.fuv_mask]) / self.flam_obs_err[self.fuv_mask]**2)
		denom = np.sum((fmodel / self.flam_obs_err[self.fuv_mask])**2)

		return num / denom


	def loglike(self, p):

		av, logz, eb, delta = p

		# load the model:
		self.model.set_spectrum(logz=logz, interpolation=self.model_interpolation)

		# just the wavelegnths where the models overlap
		overlap = np.isin(self.model.wl, self.wl_obs)
		
		if self.stellar_only:
			intrinsic_spec_all = self.model.f_stellar[overlap]
		else:
			intrinsic_spec_all = self.model.f_tot[overlap]

		intrinsic_spec = intrinsic_spec_all[self.fuv_mask]

		alam = utils.attenuation_salim_2018(wl=self.wl_obs[self.fuv_mask]/1.e4, 
			av=av, B=eb, delta=delta)
		fmodel = intrinsic_spec * np.power(10, -0.4 * alam)

		norm = self.normalize_model_to_spec(fmodel=fmodel)
		fmodel *= norm

		return -0.5 * np.sum((((fmodel - self.flam_obs[self.fuv_mask]) /\
			self.flam_obs_err[self.fuv_mask]) ** 2 + 2 * np.log(self.flam_obs_err[self.fuv_mask])))


	def prior_trans(self, up):
		""""
		av = 0 - 5
		z = (model dependent)
		"""

		uav, ulogz, ueb, udelta = up

		av = 10.0 * uav
		eb = 10.0 * ueb
		delta = 2.0 * (2.0 * udelta - 1.0)

		if self.model.model_type == 'sb99-wmbasic':
			logz = (1.6 * ulogz + 1.4) * -1.0

		return av, logz, eb, delta


	def run(self, print_progress=True):

		self.fuv_mask = self.get_fuv_continuum_mask(wl=self.wl_obs)

		self.sampler = dynesty.NestedSampler(self.loglike, 
				self.prior_trans, self.ndim, bootstrap=0)

		self.sampler.run_nested(print_progress=print_progress)


	def save_nested_results(self, ofile):

		ofile = open("{}".format(ofile),"wb")
		pickle.dump(self.sampler.results, ofile)
		ofile.close()


	def plot_fit(self, dynesty_results=None, savefile=None, save_bestfit=None, plot_steidel_windows=True):
		"""
		results - a dynesty results class object
		"""

		fig, ax = plt.subplots(figsize=(9, 5))

		if plot_steidel_windows:
			utils.plot_steidel_windows(ax=ax, wl=self.wl_obs, color='grey', alpha=0.2)

		ax.plot(self.wl_obs, self.flam_obs, drawstyle='steps', color='black',
			lw=1., label=None)

		ax.plot(self.wl_obs, self.flam_obs_err, ls='-', color='C3',
			lw=1., label=None)

		# plot the model spectrum with correct normalization:
		self.model.set_spectrum(logz=dynesty_results.logz[1], 
			interpolation=self.model_interpolation)

		# just the wavelegnths where the models overlap
		overlap = np.isin(self.model.wl, self.wl_obs)
		
		if self.stellar_only:
			intrinsic_spec_all = self.model.f_stellar[overlap]
		else:
			intrinsic_spec_all = self.model.f_tot[overlap]

		if self.fuv_mask is None:
			self.fuv_mask = self.get_fuv_continuum_mask(wl=self.wl_obs)

		# get the normalisation:
		intrinsic_spec = intrinsic_spec_all[self.fuv_mask]
		alam = utils.attenuation_salim_2018(wl=self.wl_obs[self.fuv_mask]/1.e4, 
			av=dynesty_results.av[1], B=dynesty_results.bump[1], delta=dynesty_results.delta[1])
		fmodel = intrinsic_spec * np.power(10, -0.4 * alam)
		norm = self.normalize_model_to_spec(fmodel=fmodel)
		fmodel *= norm

		resid = (fmodel - self.flam_obs[self.fuv_mask]) / self.flam_obs_err[self.fuv_mask]
		npoint = len(fmodel) - 4
		rchi2 = np.sum(resid ** 2.) / npoint

		alam_all = utils.attenuation_salim_2018(wl=self.wl_obs/1.e4, 
			av=dynesty_results.av[1], B=dynesty_results.bump[1], delta=dynesty_results.delta[1])
		fmodel = intrinsic_spec_all * np.power(10, -0.4 * alam_all)
		fmodel *= norm

		if save_bestfit:
			np.savetxt(fname=save_bestfit, X=np.column_stack((self.wl_obs, fmodel)),
				header=' wl flam')

		model_plot_range = self.wl_obs <= 2000
		ax.plot(self.wl_obs[model_plot_range], fmodel[model_plot_range], drawstyle='steps', color='C2',
			lw=2., label=None)

		# use the 1420 - 1480A region to get the y-axis normalization
		if self.wl_obs[0] < 1420:
			ok = (self.wl_obs > 1420) & (self.wl_obs < 1480)
			ax.set_ylim(0.0, 2.0 * np.median(self.flam_obs[ok]))
		else:
			ok = (self.wl_obs > 1950) & (self.wl_obs < 2000)
			ax.set_ylim(0.0, 2.0 * np.median(self.flam_obs[ok]))

		ax.set_xlabel(r'Wavelength / $\rm{\AA}$', fontsize=15)
		ax.set_ylabel(r'F$_{\lambda}$', fontsize=15)

		ax.set_title(r'log(Z$_{\star}$/Z$_{\odot}$) = %.2f; $\chi^2_{\nu}=$ %.2f' % (dynesty_results.logz[1],
			rchi2))

		in_ax = fig.add_axes([0.7,0.6,0.25,0.3])
		bins = np.arange(-5, 5, 0.2)
		in_ax.hist(resid, histtype='stepfilled', color='C2', alpha=0.7,bins=bins, density=True)
		in_ax.hist(resid, histtype='step', color='k', alpha=1.0,bins=bins, density=True)
		in_ax.set_yticks([])
		in_ax.set_xlabel('Residuals', fontsize=12)

		ax.set_xlim(1210, self.wl_obs[-1])
		plt.tight_layout()


		if savefile:
			fig.savefig(savefile)
		else:
			plt.show()

	
class BurstWeights(object):
	"""
	A fit using burst models (fitting weights at fixed age and metallicity as in Chisholm et al. 2019)
	"""

	def __init__(self, fuv_mask_type='full', dust_law='salim'):

		self.fuv_mask_type = fuv_mask_type

		# intialize the uv mask to None
		self.fuv_mask = None

		# the dust law:
		self.dust_law = dust_law

	
	def input_observations(self, wl, flam, flam_err):

		self.wl_obs = wl
		self.flam_obs = flam
		self.flam_obs_err = flam_err


	def get_fuv_continuum_mask(self, wl):
		'''
		For an input wavelegnth grid (wl), returns a mask of
		the purely stellar UV windows from Steidel + 2016
		'''
			
		m1 = (wl > 1042.) & (wl < 1080.)
		m2 = (wl > 1086.) & (wl < 1119.)
		m3 = (wl > 1123.) & (wl < 1131.)
		m4 = (wl > 1134.) & (wl < 1141.)
		m5 = (wl > 1144.) & (wl < 1186.)
		m6 = (wl > 1198.) & (wl < 1202.)
		m7 = (wl > 1221.) & (wl < 1254.)
		m8 = (wl > 1270.) & (wl < 1291.)
		m9 = (wl > 1312.) & (wl < 1328.)
		m10 = (wl > 1340.) & (wl < 1363.)
		m11 = (wl > 1373.) & (wl < 1389.)
		m12 = (wl > 1396.) & (wl < 1398.)
		m13 = (wl > 1404.) & (wl < 1521.)
		m14 = (wl > 1528.) & (wl < 1531.)
		m15 = (wl > 1536.) & (wl < 1541.)
		m16 = (wl > 1552.) & (wl < 1606.)
		m17 = (wl > 1610.) & (wl < 1657.)
		m18 = (wl > 1675.) & (wl < 1708.)
		m19 = (wl > 1711.) & (wl < 1740.)
		m20 = (wl > 1743.) & (wl < 1751.)
		m21 = (wl > 1754.) & (wl < 1845.)
		m22 = (wl > 1864.) & (wl < 1878.)
		m23 = (wl > 1885.) & (wl < 1903.)
		m24 = (wl > 1920.) & (wl < 2000.)
		m25 = (wl > 2000.) & (wl < 2300.)

		if self.fuv_mask_type == 'full': 
			mask = m7 | m8 | m9 | m10 | m11 | m12 | m13 | m14 | m15 | \
				m16 | m17 | m18 | m19 | m20 | m21 | m22 | m23 | m24
		elif self.fuv_mask_type == 'exclude_heii':
			mask = m7 | m8 | m9 | m10 | m11 | m12 | m13 | m14 | m15 | \
				m16 | m18 | m19 | m20 | m21 | m22 | m23 | m24
		elif self.fuv_mask_type == 'exclude_nv_heii':
			mask = m8 | m9 | m10 | m11 | m12 | m13 | m14 | m15 | \
				m16 | m18 | m19 | m20 | m21 | m22 | m23 | m24

		return mask


	def normalize_model_to_spec(self, fmodel):

		num = np.sum((fmodel * self.flam_obs[self.fuv_mask]) / self.flam_obs_err[self.fuv_mask]**2)
		denom = np.sum((fmodel / self.flam_obs_err[self.fuv_mask])**2)

		return num / denom

	
	def load_sb99_models(self, ages, metallicities, imf=2.3, rotation=False, 
		pixel_resolution=3.0, kernel='box', wl_grid=None, stellar_only=False):
		"""
		Load the models at their fixed metallicities
		"""

		self.model_type = 'sb99'

		self.ages = ages
		self.model_zs = metallicities

		self.ndim_sps = len(ages) * len(metallicities)

		# total dimensions of the fit (sps + dust params):
		if self.dust_law == 'salim':
			self.ndim_fit = self.ndim_sps+3
		elif self.dust_law == 'calzetti':
			self.ndim_fit = self.ndim_sps+1


		# dictionary to hold the final models:
		self.stellar_models = {}

		for age in self.ages:

			for z in self.model_zs:

				if self.model_type == 'sb99':

					_mod = fuvmod.SB99_WMBasic(imf=imf, rotation=rotation, age_myr=age, wl_grid=wl_grid, 
						pixel_resolution=pixel_resolution, kernel=kernel, sfh='burst')

					_mod.set_spectrum(logz=np.log10(z), interpolation='linear')

					self.stellar_models["{}_{}".format(str(age), str(z))] = {}
					self.stellar_models["{}_{}".format(str(age), str(z))]['age'] = age
					self.stellar_models["{}_{}".format(str(age), str(z))]['z'] = z

					if stellar_only:
						self.stellar_models["{}_{}".format(str(age), str(z))]['flam'] = _mod.f_stellar
					else:
						self.stellar_models["{}_{}".format(str(age), str(z))]['flam'] = _mod.f_tot


	def prior_trans(self, u):
		""""
		av = 0 - 5
		z = (model dependent)
		"""

		x = np.array(u)

		# flat prior between 0 - 1 for all coefficients:
		for i in range(len(u[:self.ndim_sps])):
			x[i] = 5 * (2 * u[i] - 1)

		if self.dust_law == 'salim':
			x[-3] = 2.0 * (2.0 * u[-3] - 1.0) # delta flat proper in -2 to 2
			x[-2] = 10 * u[-2] # Eb flat proper in range 0-5
			x[-1] = 10 * u[-1] # Av flat proper in range 0-5
		elif self.dust_law == 'calzetti':
			x[-1] = 10 * u[-1] # Av flat proper in range 0-5

		return x


	def loglike(self, x):

		model_flux = np.zeros_like(self.wl_obs)

		for i, (k, v) in enumerate(self.stellar_models.items()):

			model_flux += 10**x[i] * v['flam']

		intrinsic_spec = model_flux[self.fuv_mask]

		if self.dust_law == 'salim':
			alam = utils.attenuation_salim_2018(wl=self.wl_obs[self.fuv_mask]/1.e4, 
				av=x[-1], B=x[-2], delta=x[-3])
		elif self.dust_law == 'calzetti':
			alam = utils.attenuation_salim_2018(wl=self.wl_obs[self.fuv_mask]/1.e4, 
				av=x[-1], B=0.0, delta=0.0)

		fmodel = intrinsic_spec * np.power(10, -0.4 * alam)

		norm = self.normalize_model_to_spec(fmodel=fmodel)
		fmodel *= norm

		return -0.5 * np.sum((((fmodel - self.flam_obs[self.fuv_mask]) /\
			self.flam_obs_err[self.fuv_mask]) ** 2 + 2 * np.log(self.flam_obs_err[self.fuv_mask])))


	def get_chi_squared(self, x):

		model_flux = np.zeros_like(self.wl_obs)

		for i, (k, v) in enumerate(self.stellar_models.items()):

			model_flux += 10**x[i] * v['flam']

		intrinsic_spec = model_flux[self.fuv_mask]

		if self.dust_law == 'salim':
			alam = utils.attenuation_salim_2018(wl=self.wl_obs[self.fuv_mask]/1.e4, 
				av=x[-1], B=x[-2], delta=x[-3])
		elif self.dust_law == 'calzetti':
			alam = utils.attenuation_salim_2018(wl=self.wl_obs[self.fuv_mask]/1.e4, 
				av=x[-1], B=0.0, delta=0.0)

		fmodel = intrinsic_spec * np.power(10, -0.4 * alam)

		norm = self.normalize_model_to_spec(fmodel=fmodel)
		fmodel *= norm

		resid = (fmodel - self.flam_obs[self.fuv_mask]) / self.flam_obs_err[self.fuv_mask]

		return  np.sum(resid ** 2.)

	def run(self, print_progress=True, nlive=500):

		self.fuv_mask = self.get_fuv_continuum_mask(wl=self.wl_obs)

		self.sampler = dynesty.NestedSampler(self.loglike, 
				self.prior_trans, self.ndim_fit, bootstrap=0, nlive=nlive)

		self.sampler.run_nested(print_progress=print_progress)
		

	def save_nested_results(self, ofile):

		ofile = open("{}".format(ofile),"wb")
		pickle.dump(self.sampler.results, ofile)
		ofile.close()


	def estimate_maxlike_metallicity_solution(self, dynesty_results):

		samples_equal = dynesty.utils.resample_equal(dynesty_results.results['samples'], 
			dynesty_results.weights)

		# determine the minimum chi-squared from the samples:
		chi2 = np.empty(samples_equal.shape[0])

		for i in range(samples_equal.shape[0]):

			chi2[i] = self.get_chi_squared(x=samples_equal[i,:])

		maxl_params = samples_equal[np.argmin(chi2),:]

		dof = len(self.wl_obs[self.fuv_mask]) - (self.ndim_fit)
		print(min(chi2)/dof)

		xi = np.empty(self.ndim_sps)
		zi = np.empty(self.ndim_sps)

		for j, (k, v) in enumerate(self.stellar_models.items()):

			xi[j] = 10**maxl_params[j]
			zi[j] = v['z']

		return np.sum(xi * zi) / np.sum(xi)
		


	def plot_fit(self, dynesty_results=None, save_bestfit=None, savefile=None, plot_steidel_windows=True):

		if dynesty_results:
			fit_results = dynesty_results.results
		else:
			fit_results = self.sampler.results

		fig, ax = plt.subplots(figsize=(9, 5))

		if plot_steidel_windows:
			utils.plot_steidel_windows(ax=ax, wl=self.wl_obs, color='grey', alpha=0.2)

		ax.plot(self.wl_obs, self.flam_obs, drawstyle='steps', color='black',
			lw=1., label=None)

		ax.plot(self.wl_obs, self.flam_obs_err, ls='-', color='C3',
			lw=1., label=None)

		# now plot the best fitting model:
		model_flux = np.zeros_like(self.wl_obs)

		fit_w = np.exp(fit_results['logwt'] - fit_results['logz'][-1])

		for i, (k, v) in enumerate(self.stellar_models.items()):

			weight = dynesty.utils.quantile(x=fit_results['samples'][:,i], 
					q=0.5, weights=fit_w)[0]

			model_flux += 10**weight * v['flam']

		if self.dust_law == 'salim':
			av = dynesty.utils.quantile(x=fit_results['samples'][:,-1], 
						q=0.5, weights=fit_w)
			eb = dynesty.utils.quantile(x=fit_results['samples'][:,-2], 
						q=0.5, weights=fit_w)
			delta = dynesty.utils.quantile(x=fit_results['samples'][:,-3], 
						q=0.5, weights=fit_w)
			alam = utils.attenuation_salim_2018(wl=self.wl_obs/1.e4, 
				av=av, B=eb, delta=delta)
		elif self.dust_law == 'calzetti':
			av = dynesty.utils.quantile(x=fit_results['samples'][:,-1], 
						q=0.5, weights=fit_w)
			alam = utils.attenuation_salim_2018(wl=self.wl_obs/1.e4, 
				av=av, B=0.0, delta=0.0)
		
		fmodel = model_flux * np.power(10, -0.4 * alam)

		if self.fuv_mask is None:
			self.fuv_mask = self.get_fuv_continuum_mask(wl=self.wl_obs)

		norm = self.normalize_model_to_spec(fmodel=fmodel[self.fuv_mask])
		fmodel *= norm

		if save_bestfit:
			np.savetxt(fname=save_bestfit, X=np.column_stack((self.wl_obs, fmodel)),
				header=' wl flam')

		model_plot_range = self.wl_obs <= 2000
		ax.plot(self.wl_obs[model_plot_range], fmodel[model_plot_range], drawstyle='steps', color='C2',
			lw=2., label=None)

		# use the 1420 - 1480A region to get the y-axis normalization
		if self.wl_obs[0] < 1420:
			ok = (self.wl_obs > 1420) & (self.wl_obs < 1480)
			ax.set_ylim(0.0, 2.0 * np.median(self.flam_obs[ok]))
		else:
			ok = (self.wl_obs > 1950) & (self.wl_obs < 2000)
			ax.set_ylim(0.0, 2.0 * np.median(self.flam_obs[ok]))

		ax.set_xlabel(r'Wavelength / $\rm{\AA}$', fontsize=15)
		ax.set_ylabel(r'F$_{\lambda}$', fontsize=15)

		if savefile:
			fig.savefig(savefile)
		else:
			plt.show()





