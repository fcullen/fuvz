import numpy as np

def attenuation_salim_2018(wl, av, B, delta):
	"""
	Returns A(lambda) for the Salim + 2018 dust attenuation
	prescription
	""" 

	x = 1.0 / wl
	rv_calz = 4.05
	k_calz = (2.659 * (-2.156 + 1.509*x - 0.198*x**2 + 0.011*x**3)) + rv_calz

	wl0 = 0.2175
	dwl = 0.035
	d_lam = (B * np.power(wl*dwl, 2)) / (np.power(wl**2-wl0**2, 2) + np.power(wl0*dwl, 2))

	rv_mod = rv_calz / ((rv_calz + 1)*(0.44/0.55)**delta - rv_calz)

	kmod = k_calz * (rv_mod/rv_calz) * (wl/0.55)**delta + d_lam

	return (kmod * av) / rv_mod


def plot_steidel_windows(ax, wl, color, alpha):

	"""mask = (wl > 1221.) & (wl < 1254.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color='C0', alpha=alpha, lw=0.)"""
	mask = (wl > 1270.) & (wl < 1291.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1312.) & (wl < 1328.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1340.) & (wl < 1363.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1373.) & (wl < 1389.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1396.) & (wl < 1398.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1404.) & (wl < 1521.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1528.) & (wl < 1531.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1536.) & (wl < 1541.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1552.) & (wl < 1606.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1610.) & (wl < 1657.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1675.) & (wl < 1708.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1711.) & (wl < 1740.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1743.) & (wl < 1751.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1754.) & (wl < 1845.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1864.) & (wl < 1878.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1885.) & (wl < 1903.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
	mask = (wl > 1920.) & (wl < 2000.)
	ax.fill_betweenx(np.linspace(0.0, 4.0, 1000), wl[mask][0],
		wl[mask][-1], color=color, alpha=alpha, lw=0.)
