import os
from astropy.io import ascii

import numpy as np
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel

import matplotlib.pyplot as plt

import fuvz
from . import utils

import spectres as sr


class SB99_WMBasic(object):
	"""
	Starburst99 WMBasic model class
	"""

	def __init__(self, imf, rotation, age_myr, wl_grid=None, pixel_resolution=0.4, kernel='box', sfh='constant'):


		self.dir = "{}/data/sps/".format(fuvz.__path__[0])
		self.imf = imf

		self.z_names = ['001', '002', '008', '014', '040']
		self.z_vals = np.array([0.001, 0.002, 0.008, 0.014, 0.040])
		self.logz_vals = np.log10(self.z_vals)

		self.sfh = sfh

		self.model_type = 'sb99-wmbasic'

		self.wl_grid = wl_grid

		self.rotation = rotation # the geneva v00 (no rotation) or v40 (rotation) models
		self.age_myr = int(age_myr)

		self.native_resolution = 0.4 # resolution is 0.4 A
		self.pixel_resolution = pixel_resolution

		self.kernel = kernel

		# load all the model data:
		self.all_models = {}
		for zname in self.z_names:
			self.all_models[zname] = self._return_sfile_data(zname=zname)


	def print_avialiable_metallicities(self):

		print("\nThe Starburst99 WBBasic spectra cover the following metallicitiy values:")
		print("Z* = 0.001, 0.002, 0.008, 0.014, 0.040\n")


	def set_spectrum(self, logz, interpolation='linear'):
		"""
		Return a spectrum at a given metallicity, the spectra are interpolated
		in logz and then you can choose between linear or log interpolation
		in wl-flam
		"""

		# first set the metallicity as an attribute
		self.logz = logz
		self.z = 10 ** self.logz

		if self.logz < -3.0 or self.logz > -1.38:
			print("SB99 metallicity error: needs to be " +\
				"in the range -3.0 < z < -1.4\n")
			sys.exit(0)

		if self.logz in self.logz_vals:
			zname = self.z_names[np.argwhere(self.logz_vals == self.logz)[0][0]]
			self.wl, self.f_stellar,\
				self.f_tot = self._return_sfile_data(zname=zname)
		else:
			# now need to interpolate between models:
			encomp_models, encomp_vals = self._get_encompassing_models(logz=self.logz)

			if interpolation == 'linear':
				# load the model data:
				wl1,f_stellar_m1,\
					f_tot_m1 = self.all_models[encomp_models[0]]

				self.wl,f_stellar_m2,\
					f_tot_m2 = self.all_models[encomp_models[1]]

				# get the difference between models:
				f_stellar_diff = f_stellar_m1 - f_stellar_m2

				# get the fracitonal difference based on the metallicity value
				frac_diff = 1. - ((np.abs(encomp_vals[1]) - np.abs(self.logz)) / (np.abs(encomp_vals[1]) - np.abs(encomp_vals[0])))
				delta_f_stellar = frac_diff * f_stellar_diff

				self.f_stellar = f_stellar_m1 - delta_f_stellar

				# get the difference between models:
				f_tot_diff = f_tot_m1 - f_tot_m2

				# get the fracitonal difference based on the metallicity value
				frac_diff = 1. - ((np.abs(encomp_vals[1]) - np.abs(self.logz)) / (np.abs(encomp_vals[1]) - np.abs(encomp_vals[0])))

				delta_f_tot = frac_diff * f_tot_diff

				self.f_tot = f_tot_m1 - delta_f_tot
				
			elif interpolation == 'log':

				# load the model data:
				wl1,f_stellar_m1,\
					f_tot_m1 = self.all_models[encomp_models[0]]

				self.wl,f_stellar_m2,\
					f_tot_m2 = self.all_models[encomp_models[1]]

				# convert in log space:
				lwl = np.log10(wl1)
				lf_stellar_m1 = np.log10(f_stellar_m1)
				lf_stellar_m2 = np.log10(f_stellar_m2)
				lf_tot_m1 = np.log10(f_tot_m1)
				lf_tot_m2 = np.log10(f_tot_m2)


				# get the difference between models:
				f_stellar_diff = lf_stellar_m1 - lf_stellar_m2

				# get the fracitonal difference based on the metallicity value
				frac_diff = 1. - ((np.abs(encomp_vals[1]) - np.abs(self.logz)) / (np.abs(encomp_vals[1]) - np.abs(encomp_vals[0])))
				delta_f_stellar = frac_diff * f_stellar_diff

				self.f_stellar = np.power(10, lf_stellar_m1 - delta_f_stellar)

				# get the difference between models:
				f_tot_diff = lf_tot_m1 - lf_tot_m2

				# get the fracitonal difference based on the metallicity value
				frac_diff = 1. - ((np.abs(encomp_vals[1]) - np.abs(self.logz)) / (np.abs(encomp_vals[1]) - np.abs(encomp_vals[0])))

				delta_f_tot = frac_diff * f_tot_diff

				self.f_tot = np.power(10, lf_tot_m1 - delta_f_tot)


	def _return_sfile_data(self, zname):

		if self.sfh == 'constant':
			if self.rotation:
				sfile = '{}/S99-v40-Z{}-IMF{:.1f}_{:d}myr.spectrum'.format(self.dir,
						zname, self.imf, self.age_myr)
			else:
				sfile = '{}/S99-v00-Z{}-IMF{:.1f}_{:d}myr.spectrum'.format(self.dir,
						zname, self.imf, self.age_myr)
		elif self.sfh == 'burst':
			sfile = '{}/S99-v00-Z{}-burst-{:d}-myr.spectrum'.format(self.dir,
						zname, self.age_myr)

		data = ascii.read(sfile)

		_wl = data['wl']
		_f_stellar = data['f_stellar']
		_f_tot = data['f_tot']

		# convolve to a given pixel resolution
		if self.pixel_resolution and self.kernel == 'box':
			_f_stellar = convolve(_f_stellar, 
				Box1DKernel(self.pixel_resolution/self.native_resolution))
			_f_tot = convolve(_f_tot, 
				Box1DKernel(self.pixel_resolution/self.native_resolution))
		elif self.pixel_resolution and self.kernel == 'gauss':
			_f_stellar = convolve(_f_stellar, 
				Gaussian1DKernel(self.pixel_resolution/(2.355*self.native_resolution)))
			_f_tot = convolve(_f_tot, 
				Gaussian1DKernel(self.pixel_resolution/(2.355*self.native_resolution)))

		# resample onto a new wavelength grid:
		if self.wl_grid is None:
			self.wl = _wl
			self.f_stellar = _f_stellar
			self.f_tot =_f_tot
		else:
			self.wl = np.copy(self.wl_grid)
			self.f_stellar = sr.spectres(spec_wavs=_wl, 
				spec_fluxes=_f_stellar, new_wavs=self.wl)
			self.f_tot = sr.spectres(spec_wavs=_wl, 
				spec_fluxes=_f_tot, new_wavs=self.wl)

		return (self.wl, self.f_stellar, self.f_tot)


	def _get_encompassing_models(self, logz):

		if (logz > -3.0) & (logz <= -2.7):
			return ['001', '002'], [-3.0, -2.7]
		elif (logz > -2.7) & (logz <= -2.1):
			return ['002', '008'], [-2.7, -2.1]
		elif (logz > -2.1) & (logz <= -1.85):
			return ['008', '014'], [-2.1, -1.85]
		elif (logz > -1.85) & (logz <= -1.4):
			return ['014', '040'], [-1.85, -1.4]


	def plot_spectrum(self, ax=None):

		if ax is None:
			fig, ax = plt.subplots(figsize=(7, 5))
			ax.minorticks_on()

		ax.plot(self.wl, np.log10(self.f_stellar), color='grey', lw=1., ls=':', label='Stellar')
		ax.plot(self.wl, np.log10(self.f_tot), color='k', lw=1., ls='-', label='Stellar + Nebular')

		ax.set_xlabel(r'Wavelength / $\rm{\AA}$', fontsize=15)
		ax.set_ylabel(r'log(F$_{\lambda}$)', fontsize=15)

		ax.set_title(r'log(Z$_{\star}$/Z$_{\odot}$) = %.2f' % (self.logz))

		ax.legend(frameon=False, loc='upper right', fontsize=15)

		plt.show()