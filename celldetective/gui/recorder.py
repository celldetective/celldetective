import json
import os
import time
import datetime
from PyQt5.QtCore import QObject, QEvent, Qt, QTimer, QPoint
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QLineEdit, QAction
from PyQt5.QtGui import QMouseEvent, QKeyEvent
from celldetective import logger

try:
    from PyQt5.QtTest import QTest
    HAS_QTEST = True
except ImportError:
    HAS_QTEST = False
    logger.warning("PyQt5.QtTest not found. Replay functionality might be limited.")

class ActionRecorder(QObject):
    """
    Records user interactions with the GUI events and saves them to a structured log.
    """
    def __init__(self, output_dir=None):
        super().__init__()
        self.recording = False
        self.recording = False
        self.events = []
        self._connected_actions = set() # Track actions we listen to
        self.start_time = 0
        self.output_dir = output_dir
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.cleanup_old_sessions()

    def start(self):
        """Start recording events."""
        if self.recording:
            return
        self.recording = True
        self.events = []
        self.start_time = time.time()
        self.start_time = time.time()
        QApplication.instance().installEventFilter(self)
        
        # Connect to existing actions
        self._scan_and_connect_actions()
        
        logger.info("Session recording started.")

    def stop(self):
        """Stop recording events."""
        if not self.recording:
            return
        self.recording = False
        QApplication.instance().removeEventFilter(self)
        QApplication.instance().removeEventFilter(self)
        self._connected_actions.clear() # Optional: disconnect signals if needed, but risky if actions deleted
        logger.info("Session recording stopped.")

    def cleanup_old_sessions(self):
        """Delete session logs older than one week."""
        if not self.output_dir or not os.path.exists(self.output_dir):
            return

        current_time = time.time()
        one_week_seconds = 7 * 24 * 60 * 60

        try:
            for filename in os.listdir(self.output_dir):
                filepath = os.path.join(self.output_dir, filename)
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > one_week_seconds:
                        try:
                            os.remove(filepath)
                            logger.info(f"Deleted old session log: {filename}")
                        except Exception as e:
                            logger.warning(f"Failed to delete old session log {filename}: {e}")
        except Exception as e:
            logger.warning(f"Error during session log cleanup: {e}")

    def save(self, filename=None):
        """Save the recorded events to a JSON file."""
        if not self.events:
            return
            
        if filename is None:
            filename = datetime.datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S.json")
        if self.output_dir:
            filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump(self.events, f, indent=4)
            logger.info(f"Session saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save session log: {e}")

    def eventFilter(self, obj, event):
        """Captures relevant GUI events."""
        # We primarily care about events on Widgets
        if not isinstance(obj, QWidget):
            return False

        # Ignore events on internal widgets that might be noisy or irrelevant if handled by parent
        # But for robust replay, capturing specific target is usually better.

        timestamp = time.time() - self.start_time
        evt_dict = None

        if event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick):
            evt_dict = self._serialize_mouse_event(obj, event)
        elif event.type() in (QEvent.KeyPress, QEvent.KeyRelease):
            evt_dict = self._serialize_key_event(obj, event)
        elif event.type() == QEvent.ActionAdded: # New action created/added to widget
             # event.action() is only in Qt 6? Qt 5 uses specific cast or logic?
             # QActionEvent has action()
             try:
                 action = event.action()
                 if action:
                     self._connect_action(action)
             except AttributeError:
                 pass
             pass

        elif event.type() == QEvent.FocusIn:
             # Just logging focus can be helpful context, but maybe not strictly necessary for replay if we use explicit clicks
             pass

        if evt_dict:
            evt_dict['timestamp'] = timestamp
            self.events.append(evt_dict)
            
            # Periodically flush to disk could be added here if needed, but we'll rely on save()

        return False  # Let the event propagate
    
    def _scan_and_connect_actions(self):
        """Recursively find and connect to all actions in the application."""
        for widget in QApplication.topLevelWidgets():
            self._scan_widget_actions(widget)

    def _scan_widget_actions(self, widget):
        if not isinstance(widget, QWidget):
            return
            
        for action in widget.actions():
            self._connect_action(action)
            
        for child in widget.children():
            # Check if child is QWidget (children includes QObjects)
            if isinstance(child, QWidget):
                self._scan_widget_actions(child)

    def _connect_action(self, action):
        if action in self._connected_actions:
            return
        # We want to record when action is triggered
        try:
            action.triggered.connect(lambda checked=False, a=action: self._record_action_trigger(a))
            self._connected_actions.add(action)
        except Exception as e:
            # logger.warning(f"Failed to connect to action {action}: {e}")
            pass

    def _record_action_trigger(self, action):
        if not self.recording:
            return
            
        timestamp = time.time() - self.start_time
        path = self._get_action_path(action)
        
        evt_dict = {
            'type': 'ActionTriggered',
            'action_path': path,
            'text': action.text(),
            'timestamp': timestamp
        }
        self.events.append(evt_dict)

    def _get_action_path(self, action):
        # Action hierarchy is tricky. Parent can be widget or action group.
        # We use a similar strategy: WidgetPath/Action[Text]
        parent = action.parent()
        parent_path = ""
        if isinstance(parent, QWidget):
            parent_path = self._get_widget_path(parent)
        elif parent:
            parent_path = parent.objectName() # Fallback
            
        text = action.text().replace('&', '') # Remove mnemonic
        name = action.objectName()
        
        return f"{parent_path}/QAction[{text}]#{name}"

    def _get_widget_path(self, widget):
        """Generates a hierarchical path string for the widget."""
        path = []
        curr = widget
        while curr:
            name = curr.objectName()
            cls = curr.__class__.__name__
            # Identification helper: store text if it's a button/label to help disambiguate
            text_id = ""
            if hasattr(curr, 'text') and callable(curr.text):
                 t = curr.text()
                 if t and len(t) < 50: # Check length to avoid massive logs for huge text blobs
                     text_id = f"[{t}]"
            elif hasattr(curr, 'windowTitle'):
                 t = curr.windowTitle()
                 if t:
                     text_id = f"[{t}]"
            
            identifier = f"{cls}{text_id}"
            if name:
                identifier += f"#{name}"
            
            path.append(identifier)
            curr = curr.parent()
        return "/".join(reversed(path))

    def _serialize_mouse_event(self, widget, event):
        etype = {
            QEvent.MouseButtonPress: 'MouseButtonPress',
            QEvent.MouseButtonRelease: 'MouseButtonRelease',
            QEvent.MouseButtonDblClick: 'MouseButtonDblClick'
        }.get(event.type())

        return {
            'type': etype,
            'widget_path': self._get_widget_path(widget),
            'pos': (event.pos().x(), event.pos().y()),
            'global_pos': (event.globalPos().x(), event.globalPos().y()),
            'button': event.button(),
            'buttons': int(event.buttons()),
            'modifiers': int(event.modifiers())
        }

    def _serialize_key_event(self, widget, event):
        etype = {
            QEvent.KeyPress: 'KeyPress',
            QEvent.KeyRelease: 'KeyRelease'
        }.get(event.type())
        
        return {
            'type': etype,
            'widget_path': self._get_widget_path(widget),
            'key': event.key(),
            'text': event.text(),
            'modifiers': int(event.modifiers())
        }


class ActionReplayer(QObject):
    """
    Replays recorded user interactions.
    """
    def __init__(self):
        super().__init__()
        self.events = []
        self.aborted = False

    def load(self, filename):
        with open(filename, 'r') as f:
            self.events = json.load(f)

    def replay(self, speed_factor=1.0):
        """Replay the loaded events."""
        if not HAS_QTEST:
            logger.error("Cannot replay: QTest module missing.")
            return

        logger.info(f"Starting replay of {len(self.events)} events...")
        self.aborted = False
        start_time = time.time()
        
        # Sort events by timestamp just in case
        self.events.sort(key=lambda x: x['timestamp'])

        for i, event in enumerate(self.events):
            if self.aborted:
                break
            
            # Simple timing synchronization
            # Logic: sleep until valid time
            # For robustness, we might want to just execute sequentially with small delays, 
            # but respecting timestamps is better for realistic stress testing.
            
            # (Simplification: just sleep proportional to delta from previous)
            if i > 0:
                delta = event['timestamp'] - self.events[i-1]['timestamp']
                if delta > 0:
                    time.sleep(delta / speed_factor)
            
            self._execute_event(event)
            QApplication.processEvents() # processing
            
        logger.info("Replay finished.")

    def _find_action(self, path, text):
        """Find action by path/text."""
        # This scans all widgets again? Or try to resolve path?
        # Path: WidgetPath/QAction[Text]
        # logic: Find widget, then look for action.
        
        if "/QAction" not in path:
            return None
            
        widget_path, action_part = path.split("/QAction", 1)
        widget = self._find_widget(widget_path)
        if not widget:
            return None
            
        # Parse action part
        # [Text]#Name
        t_text = ""
        if "[" in action_part:
            end = action_part.rfind("]")
            t_text = action_part[1:end]
            
        # Look in widget actions
        for act in widget.actions():
            a_text = act.text().replace('&', '')
            if a_text == t_text:
                return act
                
        # Fallback: recursively search children actions
        # This handles cases where actions are children of the window/widget 
        # but not explicitly added to widget.actions() (e.g. orphan menus)
        all_actions = widget.findChildren(QAction)
        for act in all_actions:
            a_text = act.text().replace('&', '')
            if a_text == t_text:
                return act

        return None

    def _find_widget(self, path):
        """Attempts to find the widget instance in the current app matching the recorded path."""
        # This is the tricky part. 
        # Path format: WindowClass[Title]/.../WidgetClass[Text]
        
        # Heuristic:
        # 1. Get all top level widgets
        # 2. Try to match the top level
        # 3. Traverse children
        
        parts = path.split('/')
        if not parts:
            return None
        
        top_tokens = parts[0]
        
        candidates = QApplication.topLevelWidgets()
        current_widget = None
        
        # Find Top Level
        for w in candidates:
            if self._match_token(w, top_tokens):
                current_widget = w
                break
        
        if not current_widget:
            # Fallback: maybe the window structure changed slightly or it is a dialog that just popped up
            # Try to search active modal widget?
            active = QApplication.activeModalWidget()
            if active and self._match_token(active, top_tokens):
                current_widget = active
        
        if not current_widget:
            # logger.debug(f"Could not find top level widget for {top_tokens}")
            return None
            
        # Traverse downwards
        for token in parts[1:]:
            found_child = None
            # Direct children
            children = current_widget.children()
            for child in children:
                if self._match_token(child, token):
                    found_child = child
                    break
            
            if found_child:
                current_widget = found_child
            else:
                # logger.debug(f"Could not find child {token} in {current_widget}")
                return None
                
        return current_widget

    def _match_token(self, widget, token):
        """Checks if widget matches the path token."""
        # Token structure: ClassName[OptionalText]#OptionalObjName
        # Parsing crude but effective
        
        obj_name_idx = token.find('#')
        text_id_start = token.find('[')
        text_id_end = token.rfind(']')
        
        w_cls = widget.__class__.__name__
        w_obj = widget.objectName()
        
        # Parse token
        t_cls = token
        t_obj = ""
        t_text = ""
        
        if obj_name_idx != -1:
            t_obj = token[obj_name_idx+1:]
            t_cls = token[:obj_name_idx] # might still have text bracket
            
        if text_id_start != -1 and text_id_end != -1:
            # Extract text
            t_text = token[text_id_start+1:text_id_end]
            # remove text from cls if it was part of it
            if obj_name_idx == -1:
                t_cls = token[:text_id_start]
            else:
                # if # was after ], then cls is before [
                # if # was before [, impossible given our generation logic puts # at end
                if obj_name_idx > text_id_end:
                     t_cls = token[:text_id_start]
        
        # Check Class
        if t_cls != w_cls:
            return False
            
        # Check Obj Name (if recorded)
        if t_obj and t_obj != w_obj:
            return False
            
        # Check Text (if recorded and applicable)
        if t_text:
            w_text = ""
            if hasattr(widget, 'text') and callable(widget.text):
                 w_text = widget.text()
            elif hasattr(widget, 'windowTitle'):
                 w_text = widget.windowTitle()
            
            if w_text != t_text:
                # Fuzzy match? Or exact? STRICT for now, maybe loose later
                # Sometimes text changes (e.g. progress), so maybe ignore if it looks dynamic?
                # For now, require match
                return False
                
        return True

    def _execute_event(self, event):
        etype = event['type']
        
        if etype == 'ActionTriggered':
            action = self._find_action(event.get('action_path', ''), event.get('text', ''))
            if action:
                logger.info(f"Triggering action: {action.text()}")
                action.trigger()
            else:
                logger.warning(f"Could not find action: {event.get('text')}")
            return

        # For widget events, we need the widget
        widget_path = event.get('widget_path')
        if not widget_path:
            # logger.warning(f"Skipping event {etype}: No widget_path")
            return
            
        widget = self._find_widget(widget_path)
        if not widget:
            # logger.warning(f"Skipping event {etype}: Widget not found ({widget_path})")
            return

        if etype in ('MouseButtonPress', 'MouseButtonDblClick'):
            # QtTest format
            btn = Qt.MouseButton(event['button'])
            mod = Qt.KeyboardModifiers(event['modifiers'])
            pos = QPoint(*event['pos'])
            
            if etype == 'MouseButtonPress':
                try:
                     QTest.mouseClick(widget, btn, mod, pos)
                except:
                     pass
            else:
                try:
                     QTest.mouseDclick(widget, btn, mod, pos)
                except:
                     pass
                
        elif etype == 'KeyPress':
            key = event['key']
            mod = Qt.KeyboardModifiers(event['modifiers'])
            # handling special keys/chars might need more detailed mapping if strictly using QTest.keyClick
            # QTest.keyClick takes a char or a key code.
            
            # Use 'text' if available and printable
            text = event.get('text', '')
            if text and key not in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Backspace, Qt.Key_Tab, Qt.Key_Escape):
                 QTest.keyClicks(widget, text, mod, 0)
            else:
                 QTest.keyClick(widget, key, mod)

