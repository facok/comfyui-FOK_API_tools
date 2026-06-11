# -*- coding: utf-8 -*-

from .api_nodes import FOKMultiProtocolChatVisionAPI


NODE_CLASS_MAPPINGS = {
    "FOK_MultiProtocolChatVisionAPI": FOKMultiProtocolChatVisionAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FOK_MultiProtocolChatVisionAPI": "FOK Multi-Protocol Chat Vision API",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("\033[34mFOK API Tools: \033[92mLoaded custom nodes.\033[0m")
