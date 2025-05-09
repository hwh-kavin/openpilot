Project Restructure Items:

There will be 3 main views going forward:
1. BPBaseView - (Current BPPanel) which will be generated based on the config file
2. BPNestedView - which will be a version of BPNestedDialog. This will use the same groups and controls as the BPPanel.
3. BPNavBarView - which will have a NavBar on the bottom with touch-friendly buttons with icons and text.  This will use the same groups and controls as the BPPanel.

We will need to break out the groups, controls, conditions, amd whatever else into a base panel file so they can be used in all three views (Maybe bp_panel_base.h/cc).
We will need to convert the current BPPanel to use the new base class.
Each view class will inherit from the base class and add the view-specific UI components.
We can remove BPNestedDialog and BPNestedBase as they will be replaced by the new views.




We need to refactor our BP UI code to support three main views with minimal code duplication. The goals and requirements are as follows:
	1.	Three Main Views:
	•	BPBaseView:
The current BPPanel should be converted into BPBaseView. This view is generated entirely based on a JSON config file and is responsible for loading groups, controls, conditions, and handling refresh logic.
	•	BPNestedView:
This is a variant of the current BPNestedDialog. It will use the same groups and controls as BPBaseView but will be presented as a nested view (or dialog) with minor styling differences. This Might be used by both BPBaseView and BPNavBarView to hide more groups and controls behind a button.
	•	BPNavBarView:
This new view will include a bottom navigation bar with touch-friendly buttons (icons and text). Like the other views, it will load the same groups and controls as BPBaseView.
	2.	Centralization of Common Functionality:
	•	Shared Base Class (bp_panel_base.h/cc):
Break out the group/controls creation, condition handling, and refresh logic into a central base class (for example, BPPanelBase). This class will encapsulate all core functionality (loading JSON, instantiating controls, and managing state such as timers and ActivitySimulator). Please don't break the current UI layout and styling structure.
	•	View-Specific Wrappers:
Each view (BPBaseView, BPNestedView, and BPNavBarView) will either inherit from or compose an instance of BPPanelBase. They will then add their own unique UI components (e.g., the bottom navigation bar for BPNavBarView or dialog chrome for BPNestedView) without duplicating the core logic.
	3.	Activity Simulation and Timer Management:
	•	Ensure that only one widget’s activity simulation is active at any given time. The ActivitySimulator should be centralized so that if any panel requires the simulation to keep the display awake, it will use the same underlying logic.
	•	When switching between views (for example, opening a nested view from BPBaseView), the simulator should either continue running on the parent (if that meets the design) or be cleanly stopped and restarted for the child as needed. Only one requirement of activity simulation should be active at a time.
	4.	Lifecycle, Styling, and Decoupling:
	•	Remove the old BPNestedDialog and BPNestedBase classes. Their functionality should be replaced by the new BPNestedView.
	•	Ensure that the new views are decoupled from one another in terms of lifecycle. For example, the nested view should not be tightly tied as a child of BPBaseView; instead, it should be an independent widget (or dialog) that reuses the BPPanelBase core.
	•	Separate UI “chrome” (such as headers, nav bars, or back buttons) from the core panel logic. This will allow for flexible styling and easier maintenance.
	•	Consider using composition rather than inheritance if it makes it easier to “wrap” the common panel core with view-specific elements.
	5.	Future-Proofing:
	•	Design the BPPanelBase so that it is easy to add new control types or modify groups and conditions without impacting the view-specific wrappers.
	•	Maintain a clear separation of concerns between core functionality (data loading, control creation, refresh logic) and view-specific presentation (layout, navigation elements, styling).
  •	You may recommend creating new files but would like to only create files for each view-specific presentation (layout, navigation elements, styling), and one for the base class.
  •	You can also remove the

Final Detailed Instruction:

“Please refactor our BP UI code to create three main views (BPBaseView, BPNestedView, and BPNavBarView). To avoid code duplication, factor out all common logic for loading JSON configuration, creating groups, controls, conditions, and refresh logic into a base class (e.g., BPPanelBase in bp_panel_base.h/cc). Each view should either inherit from or compose BPPanelBase and add its own view-specific UI components (for example, BPNavBarView should include a bottom navigation bar with touch-friendly icons and text). Additionally, update our ActivitySimulator logic so that only one widget’s activity simulation is active at any time. The simulator should work across all views by stopping the simulation on one panel when another becomes active or when a child view closes. Remove legacy classes (like BPNestedDialog and BPNestedBase) and ensure that all lifecycles and timer events are handled cleanly to prevent UI freezing. Provide a detailed plan for organizing the files, managing lifecycles, and separating the shared core from the view-specific elements.”
