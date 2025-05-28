import numpy as np
from lifelines import KaplanMeierFitter

def switch_to_events(classes, event_times, max_times, origin_times=None, left_censored=True, FrameToMin=None, cut_observation_time=None):
	

	"""
	Converts time-to-event data into a format suitable for survival analysis, optionally adjusting for left censorship 
	and converting time units.

	This function processes event data by classifying each event based on whether it occurred or was censored by the end 
	of the observation period. It calculates the survival time for each event, taking into account the possibility of left 
	censorship and the option to convert time units (e.g., from frames to minutes).

	Parameters
	----------
	classes : array_like
		An array indicating the class of each event (e.g., 0 for event, 1 for non-event, 2 for else).
	event_times : array_like
		An array of times at which events occurred. For non-events, this might represent the time of last observation.
	max_times : array_like
		An array of maximum observation times for each event.
	origin_times : array_like, optional
		An array of origin times for each event. If None, origin times are assumed to be zero, and `left_censored` is 
		automatically set to False (default is None).
	left_censored : bool, optional
		Indicates whether to adjust for left censorship. If True, events with origin times are considered left-censored 
		if the origin time is zero (default is True).
	FrameToMin : float, optional
		A conversion factor to transform survival times from frames (or any other unit) to minutes. If None, no conversion 
		is applied (default is None).
	cut_observation_time : float or None, optional
		A cutoff time to artificially reduce the observation window and exclude late events. If None, uses all available data (default is None).

	Returns
	-------
	tuple of lists
		A tuple containing two lists: `events` and `survival_times`. `events` is a list of binary indicators (1 for event 
		occurrence, 0 for censorship), and `survival_times` is a list of survival times corresponding to each event or 
		censorship.

	Notes
	-----
	- The function assumes that `classes`, `event_times`, `max_times`, and `origin_times` (if provided) are all arrays of 
	  the same length.
	- This function is particularly useful in preparing time-to-event data for survival analysis models, especially when 
	  dealing with censored data and needing to adjust time units.

	Examples
	--------
	>>> classes = [0, 1, 0]
	>>> event_times = [5, 10, 15]
	>>> max_times = [20, 20, 20]
	>>> origin_times = [0, 0, 5]
	>>> events, survival_times = switch_to_events(classes, event_times, max_times, origin_times, FrameToMin=0.5)
	# This would process the events considering left censorship and convert survival times to minutes.
	
	"""

	events = []
	survival_times = []

	if origin_times is None:
		# then origin is zero
		origin_times = np.zeros_like(max_times)
		left_censored = False
		
	for c,t,mt,ot in zip(classes, event_times, max_times, origin_times):

		if left_censored:

			if ot>=0. and ot==ot:
				# origin time is larger than zero, no censorship
				if c==0 and t>0:
					
					delta_t = t - ot

					# Special case: observation cut at arbitrary time
					if cut_observation_time is not None:
						if t>=cut_observation_time:
							# event time larger than cut, becomes no event
							delta_t = cut_observation_time - ot # new time
							if delta_t > 0:
								events.append(0)
								survival_times.append(delta_t)
							else:
								# negative delta t, invalid cell
								pass
						else:
							# still event
							if delta_t > 0:
								events.append(1)
								survival_times.append(delta_t)	
							else:
								# negative delta t, invalid cell
								pass				
					else: 
						# standard mode
						if delta_t>0:
							events.append(1)
							survival_times.append(delta_t)
						else:
							# negative delta t, invalid cell
							pass
				elif c==1:
					delta_t = mt - ot
					if cut_observation_time is not None:
						delta_t = cut_observation_time - ot
					if delta_t>0:
						events.append(0)
						survival_times.append(delta_t)
					else:
						# negative delta t, invalid cell
						pass
				else:
					pass
			else:
				# origin time is zero, the event is left censored (we did not observe it start)
				pass

		else:
			if c==0 and t>0:
				if cut_observation_time is not None:
					if t>cut_observation_time:
						events.append(0)
						survival_times.append(cut_observation_time - ot)
					else:
						events.append(1)
						survival_times.append(t - ot)						
				else:				
					events.append(1)
					survival_times.append(t - ot)
			elif c==1:
				events.append(0)
				if cut_observation_time is not None:
					mt = cut_observation_time
				survival_times.append(mt - ot)
			else:
				pass

	if FrameToMin is not None:
		#print('convert to minutes!', FrameToMin)
		survival_times = [s*FrameToMin for s in survival_times]
	return events, survival_times

def compute_survival(df, class_of_interest, t_event, t_reference=None, FrameToMin=1, cut_observation_time=None, pairs=False):

	"""
	Computes survival analysis for a specific class of interest within a dataset, returning a fitted Kaplan-Meier 
	survival curve based on event and reference times.

	Parameters
	----------
	df : pandas.DataFrame
		The dataset containing tracking data, event times, and other relevant columns for survival analysis.
	class_of_interest : str
		The name of the column that specifies the class for which survival analysis is to be computed.
	t_event : str
		The column indicating the time of the event of interest (e.g., cell death or migration stop).
	t_reference : str or None, optional
		The reference column indicating the start or origin time for each track (e.g., detection time). If None, 
		events are not left-censored (default is None).
	FrameToMin : float, optional
		Conversion factor to scale the frame time to minutes (default is 1, assuming no scaling).
	cut_observation_time : float or None, optional
		A cutoff time to artificially reduce the observation window and exclude late events. If None, uses all available data (default is None).
	Returns
	-------
	ks : lifelines.KaplanMeierFitter or None
		A fitted Kaplan-Meier estimator object. If there are no events, returns None.

	Notes
	-----
	- The function groups the data by 'position' and 'TRACK_ID', extracting the minimum `class_of_interest` and `t_event` 
	  values for each track.
	- If `t_reference` is provided, the analysis assumes left-censoring and will use `t_reference` as the origin time for 
	  each track.
	- The function calls `switch_to_events` to determine the event occurrences and their associated survival times.
	- A Kaplan-Meier estimator (`KaplanMeierFitter`) is fitted to the data to compute the survival curve.

	Example
	-------
	>>> ks = compute_survival(df, class_of_interest="class_custom", t_event="time_custom", t_reference="t_firstdetection")
	>>> ks.plot_survival_function()
	
	"""

	cols = list(df.columns)
	assert class_of_interest in cols,"The requested class cannot be found in the dataframe..."
	assert t_event in cols,"The event time cannot be found in the dataframe..."
	left_censored = False
	first_detections = None

	if not pairs:
		groupby_cols = ['position','TRACK_ID']
	else:
		groupby_cols = ['position','REFERENCE_ID','NEIGHBOR_ID']

	classes = df.groupby(groupby_cols)[class_of_interest].min().values
	event_times = df.groupby(groupby_cols)[t_event].min().values
	max_times = df.groupby(groupby_cols)['FRAME'].max().values

	if t_reference=="0" or t_reference==0:
		t_reference = None
		left_censored = False
		first_detections = None

	if t_reference is not None:
		left_censored = True
		assert t_reference in cols,"The reference time cannot be found in the dataframe..."
		first_detections = df.groupby(groupby_cols)[t_reference].max().values
	
	events, survival_times = switch_to_events(classes, event_times, max_times, origin_times=first_detections, left_censored=left_censored, FrameToMin=FrameToMin, cut_observation_time=cut_observation_time)

	ks = KaplanMeierFitter()
	if len(events)>0:
		ks.fit(survival_times, event_observed=events)
	else:
		ks = None

	return ks

