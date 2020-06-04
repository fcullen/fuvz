from scipy.optimize import least_squares

from . import model as fuvmod
from . import utils

import numpy as np

import matplotlib.pyplot as plt

def load_sb99_models(ages, metallicities, imf=2.3, rotation=False, pixel_resolution=3.0, 
	kernel='box', wl_grid=None, stellar_only=False):
	"""
	Load the models at their fixed metallicities
	"""

	stellar_models = {}

	for age in ages:

		for z in metallicities:

			_mod = fuvmod.SB99_WMBasic(imf=imf, rotation=rotation, age_myr=age, wl_grid=wl_grid, 
				pixel_resolution=pixel_resolution, kernel=kernel, sfh='burst')

			_mod.set_spectrum(logz=np.log10(z), interpolation='linear')

			stellar_models["{}_{}".format(str(age), str(z))] = {}
			stellar_models["{}_{}".format(str(age), str(z))]['age'] = age
			stellar_models["{}_{}".format(str(age), str(z))]['z'] = z

			if stellar_only:
				stellar_models["{}_{}".format(str(age), str(z))]['flam'] = _mod.f_stellar
			else:
				stellar_models["{}_{}".format(str(age), str(z))]['flam'] = _mod.f_tot

	return stellar_models


def get_fuv_continuum_mask(wl, fuv_mask_type='full'):
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

	if fuv_mask_type == 'full': 
		mask = m7 | m8 | m9 | m10 | m11 | m12 | m13 | m14 | m15 | \
			m16 | m17 | m18 | m19 | m20 | m21 | m22 | m23 | m24
	elif fuv_mask_type == 'exclude_heii':
		mask = m7 | m8 | m9 | m10 | m11 | m12 | m13 | m14 | m15 | \
			m16 | m18 | m19 | m20 | m21 | m22 | m23 | m24
	elif fuv_mask_type == 'exclude_nv_heii':
		mask = m8 | m9 | m10 | m11 | m12 | m13 | m14 | m15 | \
			m16 | m18 | m19 | m20 | m21 | m22 | m23 | m24

	return mask


def normalize_model_to_spec(flam_model, flam_obs, flam_obs_err):

		num = np.sum((flam_model * flam_obs) / flam_obs_err**2)
		denom = np.sum((flam_model / flam_obs_err)**2)

		return num / denom


def fn_fit_residuals(x, *args, **kwargs):

	fuv_mask = get_fuv_continuum_mask(kwargs['wl'], fuv_mask_type=kwargs['fuv_mask_type'])

	wl_fit = kwargs['wl'][fuv_mask]
	flam_fit = kwargs['flam'][fuv_mask]
	flam_fit_err = kwargs['flam_err'][fuv_mask]

	model_flux = np.zeros_like(kwargs['wl'])

	for i, (k, v) in enumerate(kwargs['models'].items()):

		model_flux += x[i] * v['flam']

	flam_model = model_flux[fuv_mask]

	alam = utils.attenuation_salim_2018(wl=wl_fit/1.e4, av=x[-3], B=x[-2], delta=x[-1])
	flam_model *= np.power(10, -0.4 * alam)

	flam_model *= normalize_model_to_spec(flam_model=flam_model, flam_obs=flam_fit,
		flam_obs_err=flam_fit_err)

	return flam_model - flam_fit



def plot_fit(x, wl, flam, flam_err, stellar_models, fuv_mask_type, ndim_fit):

	fuv_mask = get_fuv_continuum_mask(wl, fuv_mask_type=fuv_mask_type)

	fig, ax = plt.subplots(figsize=(7, 5))

	ax.plot(wl, flam, color='k', ds='steps', lw=1.)
	ax.plot(wl, flam_err, color='r', ls=':', lw=1.)

	model_flux = np.zeros_like(wl)

	for i, (k, v) in enumerate(stellar_models.items()):

		model_flux += x[i] * v['flam']

	alam = utils.attenuation_salim_2018(wl=wl/1.e4, av=x[-3], B=x[-2], delta=x[-1])
	model_flux *= np.power(10, -0.4 * alam)

	flam_model_norm = model_flux[fuv_mask]
	model_flux *= normalize_model_to_spec(flam_model=flam_model_norm, flam_obs=flam[fuv_mask],
		flam_obs_err=flam_err[fuv_mask])

	resid = (model_flux[fuv_mask] - flam[fuv_mask]) / flam_err[fuv_mask]
	dof = len(wl[fuv_mask]) - ndim_fit
	rchi2 = np.sum(resid ** 2.) / dof
	print(rchi2)

	np.savetxt(fname='/Users/fcullen/Desktop/test.spec', X=np.column_stack((wl,model_flux)))

	ax.plot(wl, model_flux, color='green', ds='steps', lw=1.)

	plt.show()



def maximum_likelihood_fit(wl, flam, flam_err, ages, metallicities, fuv_mask_type='full', show_fit=False):
	"""
	Fit a spectrum using the maximum likelihood method a la Chisholm
	"""

	stellar_models = load_sb99_models(ages=ages, metallicities=metallicities,
		imf=2.3, rotation=False, pixel_resolution=3.0, kernel='box', wl_grid=wl, stellar_only=False)

	ndim_sps = ages.shape[0] * metallicities.shape[0]

	x0 = np.ones(ndim_sps + 3)

	inps = {'wl': wl, 'flam': flam, 'flam_err': flam_err,
		'fuv_mask_type': fuv_mask_type, 'models': stellar_models}

	bounds_lower = np.zeros_like(x0)
	bounds_lower[-1] = -5.0

	bounds_upper= np.ones_like(bounds_lower) * np.inf

	result = least_squares(fun=fn_fit_residuals, x0=x0, kwargs=inps, bounds=(bounds_lower, bounds_upper))

	xi = np.empty(ndim_sps)
	zi = np.empty(ndim_sps)
	ai = np.empty(ndim_sps)

	for i, (k, v) in enumerate(stellar_models.items()):

		xi[i] = result['x'][i]
		zi[i] = v['z']
		ai[i] = v['age']

	z = np.sum(xi * zi) / np.sum(xi)
	age = np.sum(xi * ai) / np.sum(xi)

	if show_fit:

		plot_fit(x=result['x'], wl=wl, flam=flam, flam_err=flam_err, 
			stellar_models=stellar_models, fuv_mask_type=fuv_mask_type, ndim_fit=ndim_sps+3)

	return result, z, age