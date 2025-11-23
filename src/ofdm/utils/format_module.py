"""
================================================================================
FILE FORMAT RECOVERY MODULE
================================================================================
Detects and repairs file format headers after transmission errors.
Useful when files are corrupted during wireless transmission but data is intact.

Supported Formats:
- PNG (Image)
- JPEG (Image)
- PDF (Document)
- GIF (Image)
- MP3 (Audio)
- WAV (Audio)
- ZIP (Archive)
- TXT (Text - UTF-8 validation)

Usage:
    from format_module import FileFormatFixer
    
    fixer = FileFormatFixer()
    fixer.auto_fix('corrupted_file.png', 'fixed_file.png')
================================================================================
"""

import struct
from pathlib import Path
from typing import Optional, Tuple

class FileFormatFixer:
    """Detect and repair file format headers"""
    
    # File Signatures (Magic Numbers)
    SIGNATURES = {
        'png': {
            'header': b'\x89PNG\r\n\x1a\n',
            'offset': 0,
            'extension': '.png'
        },
        'jpeg': {
            'header': b'\xFF\xD8\xFF',
            'offset': 0,
            'extension': '.jpg'
        },
        'gif': {
            'header': b'GIF89a',
            'offset': 0,
            'extension': '.gif'
        },
        'pdf': {
            'header': b'%PDF-',
            'offset': 0,
            'extension': '.pdf'
        },
        'zip': {
            'header': b'PK\x03\x04',
            'offset': 0,
            'extension': '.zip'
        },
        'mp3': {
            'header': b'ID3',
            'offset': 0,
            'extension': '.mp3'
        },
        'wav': {
            'header': b'RIFF',
            'offset': 0,
            'secondary': b'WAVE',
            'secondary_offset': 8,
            'extension': '.wav'
        }
    }
    
    def __init__(self):
        self.verbose = True
    
    def detect_format(self, file_path: str) -> Optional[str]:
        """
        Detect file format by reading first bytes
        
        Returns:
            Format name (e.g., 'png', 'jpeg') or None
        """
        path = Path(file_path)
        if not path.exists():
            if self.verbose:
                print(f"❌ File not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                first_bytes = f.read(64)  # Read enough to check all signatures
            
            for fmt_name, sig_info in self.SIGNATURES.items():
                header = sig_info['header']
                offset = sig_info['offset']
                
                if len(first_bytes) >= offset + len(header):
                    if first_bytes[offset:offset+len(header)] == header:
                        # Check secondary signature if exists (e.g., WAV)
                        if 'secondary' in sig_info:
                            sec_offset = sig_info['secondary_offset']
                            sec_header = sig_info['secondary']
                            if len(first_bytes) >= sec_offset + len(sec_header):
                                if first_bytes[sec_offset:sec_offset+len(sec_header)] == sec_header:
                                    return fmt_name
                        else:
                            return fmt_name
            
            # Check if it's valid UTF-8 text
            try:
                first_bytes.decode('utf-8')
                return 'text'
            except UnicodeDecodeError:
                pass
            
            if self.verbose:
                print(f"⚠️  Unknown file format: {file_path}")
            return None
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error detecting format: {e}")
            return None
    
    def check_header(self, file_path: str, expected_format: str) -> bool:
        """
        Check if file has correct header for expected format
        
        Returns:
            True if header is correct, False otherwise
        """
        detected = self.detect_format(file_path)
        return detected == expected_format
    
    def fix_header(self, file_path: str, target_format: str, output_path: Optional[str] = None) -> bool:
        """
        Fix file header by prepending correct magic bytes
        
        Args:
            file_path: Input file path
            target_format: Format to fix to (e.g., 'png', 'jpeg')
            output_path: Output file path (if None, overwrites original)
            
        Returns:
            True if successful, False otherwise
        """
        if target_format not in self.SIGNATURES:
            if self.verbose:
                print(f"❌ Unsupported format: {target_format}")
            return False
        
        path = Path(file_path)
        if not path.exists():
            if self.verbose:
                print(f"❌ File not found: {file_path}")
            return False
        
        sig_info = self.SIGNATURES[target_format]
        correct_header = sig_info['header']
        
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Check if header is already correct
            if content.startswith(correct_header):
                if self.verbose:
                    print(f"✅ Header already correct for {target_format}")
                return True
            
            # Find where actual data starts (skip corrupted header)
            # For most formats, we can try to find the correct header in the file
            header_pos = content.find(correct_header)
            
            if header_pos > 0:
                # Header found later in file, extract from there
                if self.verbose:
                    print(f"🔧 Found correct header at offset {header_pos}, trimming")
                fixed_content = content[header_pos:]
            elif header_pos == 0:
                # Already has correct header
                fixed_content = content
            else:
                # Header not found, prepend it
                if self.verbose:
                    print(f"🔧 Prepending {target_format} header")
                fixed_content = correct_header + content
            
            # Write fixed file
            output = output_path if output_path else file_path
            with open(output, 'wb') as f:
                f.write(fixed_content)
            
            if self.verbose:
                print(f"✅ Fixed header: {output}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error fixing header: {e}")
            return False
    
    def auto_fix(self, file_path: str, output_path: Optional[str] = None, 
                  expected_format: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Automatically detect and fix file format
        
        Args:
            file_path: Input file
            output_path: Output file (if None, overwrites original)
            expected_format: Expected format (if None, tries to detect from extension)
            
        Returns:
            (success, detected_format)
        """
        path = Path(file_path)
        
        # Determine expected format
        if expected_format is None:
            # Try to infer from file extension
            ext = path.suffix.lower().lstrip('.')
            
            # Map extensions to format names
            ext_map = {
                'png': 'png',
                'jpg': 'jpeg',
                'jpeg': 'jpeg',
                'gif': 'gif',
                'pdf': 'pdf',
                'zip': 'zip',
                'mp3': 'mp3',
                'wav': 'wav',
                'txt': 'text'
            }
            
            expected_format = ext_map.get(ext)
            
            if expected_format is None:
                if self.verbose:
                    print(f"⚠️  Cannot infer format from extension: {ext}")
                return False, None
        
        # Detect current format
        detected = self.detect_format(file_path)
        
        if detected == expected_format:
            if self.verbose:
                print(f"✅ File is already valid {expected_format}")
            return True, detected
        
        # Try to fix
        if self.verbose:
            print(f"🔧 Attempting to fix {expected_format} header...")
        
        success = self.fix_header(file_path, expected_format, output_path)
        
        return success, expected_format
    
    def validate_image(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate and attempt to fix image files
        
        Returns:
            (is_valid, error_message)
        """
        try:
            from PIL import Image
            
            # Try to open with PIL
            img = Image.open(file_path)
            img.verify()
            
            return True, None
            
        except Exception as e:
            error_msg = str(e)
            
            # Try to auto-fix
            if self.verbose:
                print(f"⚠️  Image validation failed: {error_msg}")
                print(f"🔧 Attempting auto-fix...")
            
            success, fmt = self.auto_fix(file_path)
            
            if success:
                # Try validating again
                try:
                    img = Image.open(file_path)
                    img.verify()
                    return True, None
                except Exception as e2:
                    return False, f"Fix failed: {str(e2)}"
            else:
                return False, error_msg


# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def fix_received_file(file_path: str, expected_extension: str = None) -> bool:
    """
    Convenience function to fix a received file
    
    Usage:
        fix_received_file('received_image.png')
        fix_received_file('received_doc.pdf', expected_extension='pdf')
    """
    fixer = FileFormatFixer()
    
    if expected_extension:
        # Map extension to format
        ext_map = {
            '.png': 'png',
            '.jpg': 'jpeg',
            '.jpeg': 'jpeg',
            '.gif': 'gif',
            '.pdf': 'pdf',
            '.zip': 'zip',
            '.mp3': 'mp3',
            '.wav': 'wav'
        }
        expected_extension = expected_extension.lower()
        if not expected_extension.startswith('.'):
            expected_extension = '.' + expected_extension
        
        fmt = ext_map.get(expected_extension)
        if fmt:
            success, _ = fixer.auto_fix(file_path, expected_format=fmt)
            return success
    
    # Auto-detect
    success, _ = fixer.auto_fix(file_path)
    return success


# ================================================================================
# MAIN (Testing)
# ================================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python format_module.py <file_path> [expected_format]")
        print("\nExample:")
        print("  python format_module.py corrupted_image.png")
        print("  python format_module.py received_doc.pdf pdf")
        sys.exit(1)
    
    file_path = sys.argv[1]
    expected_format = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("="*60)
    print(" "*15 + "FILE FORMAT FIXER")
    print("="*60)
    
    fixer = FileFormatFixer()
    
    print(f"\n📁 File: {file_path}")
    
    # Detect
    detected = fixer.detect_format(file_path)
    print(f"🔍 Detected Format: {detected if detected else 'Unknown'}")
    
    # Auto-fix
    success, fmt = fixer.auto_fix(file_path, expected_format=expected_format)
    
    if success:
        print(f"\n✅ File is valid {fmt}!")
    else:
        print(f"\n❌ Could not fix file")
    
    print("="*60)
