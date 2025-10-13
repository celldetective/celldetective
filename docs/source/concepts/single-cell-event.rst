Single-cell event
-----------------

Single cell measurements are performed instantaneously, one image at a time, implying that there is no integration of time or description of dynamic phenomena yet. The time-dependence emerges naturally when these measurements are represented as single cell signals, *i.e.* 1D timeseries, over which we can hope to detect transitions characterizing the dynamic biological phenomena of interest.

Our formulation for this problem is that cells can be classified into three categories with respect to an event:

#. Cells that exhibit an event of interest during the observation window: class “event”,

#. Cells that do not exhibit it: class “no event”,

#. Cells that either exhibited the event before the observation started or else: class “else”.

Cells belonging to the first class, can be associated with a time :math:`t_{\textrm{event}}`, interpreted as the time at which the event starts. By convention, one may take the onset of the event, to be able to perform survival studies.