"""
Translation utilities for the multilingual legal chatbot.
This module handles all translation-related functionality including language detection,
text translation, and language management.
"""

from deep_translator import GoogleTranslator
from typing import Dict, Optional, Tuple
import logging

# Language codes and names
LANGUAGES = {
    'en': 'english',
    'es': 'spanish',
    'fr': 'french',
    'de': 'german',
    'it': 'italian',
    'pt': 'portuguese',
    'ru': 'russian',
    'ja': 'japanese',
    'ko': 'korean',
    'zh': 'chinese',
    'ar': 'arabic',
    'hi': 'hindi',
    'tr': 'turkish',
    'pl': 'polish',
    'nl': 'dutch',
    'sv': 'swedish',
    'da': 'danish',
    'no': 'norwegian',
    'fi': 'finnish',
    'he': 'hebrew',
    'th': 'thai',
    'vi': 'vietnamese',
    'id': 'indonesian',
    'ta': 'tamil',
    'te': 'telugu',
    'bn': 'bengali',
    'ur': 'urdu',
    'ml': 'malayalam',
    'kn': 'kannada',
    'gu': 'gujarati',
    'pa': 'punjabi'
}

# Popular languages for quick selection
POPULAR_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'bn': 'Bengali',
    'ur': 'Urdu',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'es': 'Spanish',
    'fr': 'French'
}

# Native language names
NATIVE_NAMES = {
    'en': 'English',
    'es': 'Español',
    'fr': 'Français',
    'de': 'Deutsch',
    'it': 'Italiano',
    'pt': 'Português',
    'ru': 'Русский',
    'ja': '日本語',
    'ko': '한국어',
    'zh': '中文',
    'ar': 'العربية',
    'hi': 'हिन्दी',
    'tr': 'Türkçe',
    'pl': 'Polski',
    'nl': 'Nederlands',
    'sv': 'Svenska',
    'da': 'Dansk',
    'no': 'Norsk',
    'fi': 'Suomi',
    'he': 'עברית',
    'th': 'ไทย',
    'vi': 'Tiếng Việt',
    'id': 'Bahasa Indonesia',
    'ta': 'தமிழ்',
    'te': 'తెలుగు',
    'bn': 'বাংলা',
    'ur': 'اردو',
    'ml': 'മലയാളം',
    'kn': 'ಕನ್ನಡ',
    'gu': 'ગુજરાતી',
    'pa': 'ਪੰਜਾਬੀ'
}

class TranslationManager:
    """
    Handles translation operations for the legal chatbot.
    Provides language detection, text translation, and language management utilities.
    """
    
    def __init__(self, enable_logging: bool = True):
        """
        Initialize the translation manager.
        
        Args:
            enable_logging (bool): Enable logging for translation operations
        """
        self.translator = None
        self.available_languages = LANGUAGES
        self.popular_languages = POPULAR_LANGUAGES
        self.native_names = NATIVE_NAMES
        self.enable_logging = enable_logging
        
        # Set up logging
        if self.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            
        self._initialize_translator()
    
    def _initialize_translator(self) -> None:
        """Initialize translator with error handling."""
        try:
            # Test the translator with a simple translation
            test_result = GoogleTranslator(source='auto', target='es').translate("test")
            if not test_result:
                raise Exception("Translator initialization test failed")
            
            self.translator = GoogleTranslator
            
            if self.enable_logging:
                self.logger.info("Translation service initialized successfully")
                
        except Exception as e:
            error_msg = f"Translation service initialization failed: {str(e)}"
            
            if self.enable_logging:
                self.logger.error(error_msg)
                
            self.translator = None
            print(f"❌ {error_msg}")
    
    def is_available(self) -> bool:
        """Check if translation service is available."""
        return self.translator is not None
    
    def get_languages(self) -> Dict[str, str]:
        """Get available languages."""
        return self.available_languages.copy()
    
    def get_popular_languages(self) -> Dict[str, str]:
        """Get popular languages for quick selection."""
        return self.popular_languages.copy()
    
    def get_native_language_name(self, lang_code: str) -> str:
        """
        Get native name of language.
        
        Args:
            lang_code (str): Language code (e.g., 'hi', 'te', 'en')
            
        Returns:
            str: Native name of the language
        """
        return self.native_names.get(lang_code, LANGUAGES.get(lang_code, 'Unknown'))
    
    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect language of input text.
        
        Args:
            text (str): Text to detect language for
            
        Returns:
            Optional[str]: Detected language code or None if detection fails
        """
        if not text or len(text.strip()) < 3:
            return None
            
        if not self.is_available():
            return None
        
        try:
            # Use GoogleTranslator to detect language
            translator = self.translator(source='auto', target='en')
            detected = translator.detect(text)
            
            if self.enable_logging:
                self.logger.info(f"Detected language: {detected} for text: {text[:50]}...")
            
            return detected
            
        except Exception as e:
            error_msg = f"Language detection failed: {str(e)}"
            
            if self.enable_logging:
                self.logger.warning(error_msg)
                
            print(f"⚠️ {error_msg}")
            return None
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text from source to target language.
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code
            
        Returns:
            Optional[str]: Translated text or original text if translation fails
        """
        if not text or not text.strip() or source_lang == target_lang:
            return text
            
        if not self.is_available():
            return text
        
        try:
            if source_lang == "auto":
                translator = self.translator(source='auto', target=target_lang)
            else:
                translator = self.translator(source=source_lang, target=target_lang)
            
            # Split long text to avoid hitting character limits
            if len(text) > 4000:  # Safe limit for most translations
                chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
                translated_chunks = []
                for chunk in chunks:
                    translated = translator.translate(chunk)
                    translated_chunks.append(translated)
                result = ' '.join(translated_chunks)
            else:
                result = translator.translate(text)
            
            if self.enable_logging:
                self.logger.info(f"Translated text from {source_lang} to {target_lang}")
                
            return result
            
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            
            if self.enable_logging:
                self.logger.error(error_msg)
                
            print(f"❌ {error_msg}")
            return text
    
    def translate_with_detection(self, text: str, target_lang: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        Translate text with automatic language detection.
        
        Args:
            text (str): Text to translate
            target_lang (str): Target language code
            
        Returns:
            Tuple[Optional[str], Optional[str], str]: (translated_text, detected_lang, info_message)
        """
        if not text or not text.strip():
            return text, None, ""
        
        detected_lang = self.detect_language(text)
        
        if detected_lang and detected_lang != target_lang:
            translated_text = self.translate_text(text, detected_lang, target_lang)
            detected_lang_name = self.available_languages.get(detected_lang, detected_lang)
            target_lang_name = self.available_languages.get(target_lang, target_lang)
            info_message = f"Detected {detected_lang_name} → Translated to {target_lang_name}"
            return translated_text, detected_lang, info_message
        else:
            return text, detected_lang, ""
    
    def get_language_options_for_api(self) -> Dict[str, Dict[str, str]]:
        """
        Get formatted language options for API responses.
        
        Returns:
            Dict: Language information for API consumption
        """
        language_options = {}
        
        for code, name in self.available_languages.items():
            native_name = self.get_native_language_name(code)
            language_options[code] = {
                'code': code,
                'name': name.title(),
                'native_name': native_name,
                'is_popular': code in self.popular_languages
            }
        
        return language_options
    
    def get_status_message(self) -> str:
        """Get translation service status message."""
        return "Active" if self.is_available() else "Inactive"
    
    def get_supported_languages_for_ui(self) -> list:
        """
        Get list of supported languages formatted for React UI.
        
        Returns:
            list: List of language objects with code, name, and native_name
        """
        return [
            {
                'code': code,
                'name': name.title(),
                'native_name': self.get_native_language_name(code),
                'is_popular': code in self.popular_languages
            }
            for code, name in self.available_languages.items()
        ]

def test_translation_service() -> bool:
    """
    Test the translation service functionality.
    
    Returns:
        bool: True if test passes, False otherwise
    """
    try:
        test_result = GoogleTranslator(source='auto', target='es').translate("Hello")
        print(f"Translation test successful: 'Hello' -> '{test_result}'")
        return True
    except Exception as e:
        print(f"Translation test failed: {str(e)}")
        return False

def initialize_translation_manager() -> TranslationManager:
    """
    Initialize and return a translation manager instance.
    
    Returns:
        TranslationManager: Configured translation manager instance
    """
    return TranslationManager()
