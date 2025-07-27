from typing import List
from dataclasses import dataclass
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)

class VisualMappingDisplay:
    def __init__(self):
        self.confidence_symbols = {
            'high': '🟢',      # >= 0.7
            'medium': '🟡',    # 0.5 - 0.7
            'low': '🔴'        # < 0.5
        }
        
        # Color schemes
        self.colors = {
            'high': Fore.GREEN,
            'medium': Fore.YELLOW,
            'low': Fore.RED,
            'unmapped': Fore.CYAN,
            'header': Fore.MAGENTA + Style.BRIGHT,
            'title': Fore.BLUE + Style.BRIGHT,
            'success': Fore.GREEN + Style.BRIGHT,
            'warning': Fore.YELLOW + Style.BRIGHT,
            'error': Fore.RED + Style.BRIGHT,
            'info': Fore.CYAN + Style.BRIGHT
        }
    
    def get_confidence_color_and_symbol(self, confidence: float) -> tuple:
        """Get color and symbol based on confidence level"""
        if confidence >= 0.7:
            return self.colors['high'], self.confidence_symbols['high']
        elif confidence >= 0.5:
            return self.colors['medium'], self.confidence_symbols['medium']
        else:
            return self.colors['low'], self.confidence_symbols['low']
    
    def print_fancy_header(self, title: str, width: int = 80):
        """Print a fancy header with borders"""
        print(f"{self.colors['title']}{'═' * width}")
        print(f"{'║':<1} {title:^{width-4}} {'║':>1}")
        print(f"{'═' * width}{Style.RESET_ALL}")
    
    def print_section_header(self, title: str, emoji: str = "📋"):
        """Print a section header"""
        print(f"\n{self.colors['header']}{emoji} {title}{Style.RESET_ALL}")
        print(f"{self.colors['header']}{'─' * (len(title) + 4)}{Style.RESET_ALL}")
    
    def display_beautiful_mappings(self, mappings, source_fields, target_fields, show_details=False, title="🔗 SMART DATA MAPPING RESULTS"):
        """Display mappings with beautiful formatting and colors"""
        
        self.print_fancy_header(title, 85)
        
        if not mappings:
            print(f"\n{self.colors['error']}❌ No mappings found!{Style.RESET_ALL}")
            return
        
        # Get mapped fields
        mapped_source_fields = {m.source_field for m in mappings}
        mapped_target_fields = {m.target_field for m in mappings}
        
        # Display successful mappings with beautiful formatting
        self.print_section_header(f"MAPPING SUGGESTIONS ({len(mappings)} found)", "✨")
        
        for i, mapping in enumerate(mappings, 1):
            color, symbol = self.get_confidence_color_and_symbol(mapping.confidence_score)
            
            # Create a beautiful mapping line
            print(f"{Fore.WHITE}{i:2d}.{Style.RESET_ALL} "
                  f"{Fore.BLUE + Style.BRIGHT}{mapping.source_field:<22}{Style.RESET_ALL} "
                  f"{Fore.MAGENTA}→{Style.RESET_ALL} "
                  f"{Fore.GREEN + Style.BRIGHT}{mapping.target_field:<22}{Style.RESET_ALL} "
                  f"{color}{symbol} {mapping.confidence_score:.3f}{Style.RESET_ALL}")
            
            if show_details:
                print(f"   {Fore.CYAN}├─ Type: {mapping.similarity_type}{Style.RESET_ALL}")
                print(f"   {Fore.CYAN}└─ Reason: {mapping.reasoning}{Style.RESET_ALL}")
                print()
        
        # Display unmapped fields
        unmapped_source = [f for f in source_fields if f not in mapped_source_fields]
        unmapped_target = [f for f in target_fields if f not in mapped_target_fields]
        
        if unmapped_source or unmapped_target:
            self.print_section_header("UNMAPPED FIELDS", "⚠️")
            
            if unmapped_source:
                print(f"\n{self.colors['warning']}📤 Source fields not mapped:{Style.RESET_ALL}")
                for i, field in enumerate(unmapped_source, 1):
                    print(f"   {self.colors['unmapped']}{i}. {field}{Style.RESET_ALL}")
            
            if unmapped_target:
                print(f"\n{self.colors['warning']}📥 Target fields not mapped:{Style.RESET_ALL}")
                for i, field in enumerate(unmapped_target, 1):
                    print(f"   {self.colors['unmapped']}{i}. {field}{Style.RESET_ALL}")
        
        # Beautiful summary with progress bar
        self.print_section_header("MAPPING SUMMARY", "📊")
        
        # Calculate coverage based on unique source fields mapped
        coverage_percent = len(mapped_source_fields) / len(source_fields) * 100 if source_fields else 0
        progress_bar = self._create_progress_bar(coverage_percent)
        
        print(f"\n{Fore.WHITE}📈 Source Field Coverage: {self.colors['success']}{len(mapped_source_fields)}/{len(source_fields)} "
              f"({coverage_percent:.1f}%){Style.RESET_ALL} of source fields have at least one suggestion.")
        print(f"   {progress_bar}")
        
        print(f"\n{Fore.WHITE}📋 Field counts:{Style.RESET_ALL}")
        print(f"   {Fore.BLUE}Source fields:     {len(source_fields)}{Style.RESET_ALL}")
        print(f"   {Fore.GREEN}Target fields:     {len(target_fields)}{Style.RESET_ALL}")
        print(f"   {self.colors['success']}Mapped pairs:      {len(mappings)}{Style.RESET_ALL}")
        print(f"   {self.colors['warning']}Unmapped source:   {len(unmapped_source)}{Style.RESET_ALL}")
        print(f"   {self.colors['warning']}Unmapped target:   {len(unmapped_target)}{Style.RESET_ALL}")
        
        # Confidence breakdown with visual indicators
        high_conf = len([m for m in mappings if m.confidence_score >= 0.7])
        medium_conf = len([m for m in mappings if 0.5 <= m.confidence_score < 0.7])
        low_conf = len([m for m in mappings if m.confidence_score < 0.5])
        
        print(f"\n{Fore.WHITE}🎯 Confidence levels:{Style.RESET_ALL}")
        print(f"   {self.colors['high']}🟢 High (≥0.7):     {high_conf} mappings{Style.RESET_ALL}")
        print(f"   {self.colors['medium']}🟡 Medium (0.5-0.7): {medium_conf} mappings{Style.RESET_ALL}")
        print(f"   {self.colors['low']}🔴 Low (<0.5):      {low_conf} mappings{Style.RESET_ALL}")
        
        print(f"\n{self.colors['title']}{'═' * 85}{Style.RESET_ALL}")
    
    def _create_progress_bar(self, percentage: float, width: int = 50) -> str:
        """Create a visual progress bar"""
        filled = int(width * percentage / 100)
        bar = '█' * filled + '░' * (width - filled)
        
        if percentage >= 80:
            color = self.colors['high']
        elif percentage >= 60:
            color = self.colors['medium']
        else:
            color = self.colors['low']
        
        return f"{color}[{bar}] {percentage:.1f}%{Style.RESET_ALL}"
    
    def display_compact_beautiful(self, mappings, source_fields, target_fields):
        """Display mappings in a compact but beautiful format"""
        
        print(f"\n{Back.BLUE + Fore.WHITE + Style.BRIGHT} 🔗 FIELD MAPPINGS {Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'─' * 50}{Style.RESET_ALL}")
        
        if not mappings:
            print(f"{self.colors['error']}❌ No mappings found{Style.RESET_ALL}")
            return
        
        # Beautiful mapping list
        for i, mapping in enumerate(mappings, 1):
            color, symbol = self.get_confidence_color_and_symbol(mapping.confidence_score)
            arrow = f"{Fore.MAGENTA + Style.BRIGHT}→{Style.RESET_ALL}"
            
            print(f"{Fore.WHITE}{i:2d}.{Style.RESET_ALL} "
                  f"{Fore.CYAN + Style.BRIGHT}{mapping.source_field:<20}{Style.RESET_ALL} "
                  f"{arrow} "
                  f"{Fore.GREEN + Style.BRIGHT}{mapping.target_field:<20}{Style.RESET_ALL} "
                  f"{color}{symbol}{Style.RESET_ALL}")
        
        # Unmapped fields in a compact format
        mapped_source = [m.source_field for m in mappings]
        mapped_target = [m.target_field for m in mappings]
        unmapped_source = [f for f in source_fields if f not in mapped_source]
        unmapped_target = [f for f in target_fields if f not in mapped_target]
        
        if unmapped_source:
            print(f"\n{self.colors['warning']}⚠️  Unmapped source: {Style.RESET_ALL}"
                  f"{self.colors['unmapped']}{', '.join(unmapped_source)}{Style.RESET_ALL}")
        
        if unmapped_target:
            print(f"{self.colors['warning']}⚠️  Unmapped target: {Style.RESET_ALL}"
                  f"{self.colors['unmapped']}{', '.join(unmapped_target)}{Style.RESET_ALL}")
        
        # Quick stats
        coverage = len(mappings)/len(source_fields)*100
        progress_bar = self._create_progress_bar(coverage, 30)
        print(f"\n{Fore.WHITE}📊 Coverage: {progress_bar}{Style.RESET_ALL}")
    
    def display_mapping_cards(self, mappings, source_fields, target_fields):
        """Display mappings as beautiful cards"""
        
        self.print_fancy_header("🎴 MAPPING CARDS VIEW", 70)
        
        if not mappings:
            print(f"\n{self.colors['error']}❌ No mappings found!{Style.RESET_ALL}")
            return
        
        for i, mapping in enumerate(mappings, 1):
            color, symbol = self.get_confidence_color_and_symbol(mapping.confidence_score)
            
            print(f"\n{color}╭{'─' * 60}╮{Style.RESET_ALL}")
            print(f"{color}│{Style.RESET_ALL} {symbol} "
                  f"{Fore.WHITE + Style.BRIGHT}Mapping #{i}{Style.RESET_ALL}"
                  f"{color}{' ' * (60 - len(f'Mapping #{i}') - 4)}│{Style.RESET_ALL}")
            print(f"{color}├{'─' * 60}┤{Style.RESET_ALL}")
            print(f"{color}│{Style.RESET_ALL} "
                  f"{Fore.BLUE + Style.BRIGHT}Source:{Style.RESET_ALL} {mapping.source_field:<20} "
                  f"{color}{' ' * (60 - len(mapping.source_field) - 8)}│{Style.RESET_ALL}")
            print(f"{color}│{Style.RESET_ALL} "
                  f"{Fore.GREEN + Style.BRIGHT}Target:{Style.RESET_ALL} {mapping.target_field:<20} "
                  f"{color}{' ' * (60 - len(mapping.target_field) - 8)}│{Style.RESET_ALL}")
            print(f"{color}│{Style.RESET_ALL} "
                  f"{Fore.YELLOW + Style.BRIGHT}Confidence:{Style.RESET_ALL} {mapping.confidence_score:.3f} "
                  f"({mapping.similarity_type})"
                  f"{color}{' ' * (60 - len(f'{mapping.confidence_score:.3f} ({mapping.similarity_type})') - 12)}│{Style.RESET_ALL}")
            print(f"{color}╰{'─' * 60}╯{Style.RESET_ALL}")
        
        # Show unmapped in a box
        mapped_source = [m.source_field for m in mappings]
        mapped_target = [m.target_field for m in mappings]
        unmapped_source = [f for f in source_fields if f not in mapped_source]
        unmapped_target = [f for f in target_fields if f not in mapped_target]
        
        if unmapped_source or unmapped_target:
            print(f"\n{self.colors['warning']}╭{'─' * 60}╮{Style.RESET_ALL}")
            print(f"{self.colors['warning']}│{Style.RESET_ALL} "
                  f"{Fore.YELLOW + Style.BRIGHT}⚠️  UNMAPPED FIELDS{Style.RESET_ALL}"
                  f"{self.colors['warning']}{' ' * (60 - 17)}│{Style.RESET_ALL}")
            print(f"{self.colors['warning']}├{'─' * 60}┤{Style.RESET_ALL}")
            
            if unmapped_source:
                print(f"{self.colors['warning']}│{Style.RESET_ALL} "
                      f"{Fore.RED}Source:{Style.RESET_ALL} {', '.join(unmapped_source):<46} "
                      f"{self.colors['warning']}│{Style.RESET_ALL}")
            
            if unmapped_target:
                print(f"{self.colors['warning']}│{Style.RESET_ALL} "
                      f"{Fore.RED}Target:{Style.RESET_ALL} {', '.join(unmapped_target):<46} "
                      f"{self.colors['warning']}│{Style.RESET_ALL}")
            
            print(f"{self.colors['warning']}╰{'─' * 60}╯{Style.RESET_ALL}")

# Enhanced demo function with beautiful displays
def demo_beautiful_display():
    # Import your existing mapper
    try:
        from main import SmartDataMapper
    except ImportError:
        print("Please make sure your SmartDataMapper is in main.py")
        return
    
    # Initialize
    mapper = SmartDataMapper()
    display = VisualMappingDisplay()
    
    # Your demo data
    source_schema = [
        "customer_id", "cust_name", "email_addr", "phone_num", 
        "billing_address", "created_date", "account_status", "total_amount"
    ]
    
    target_schema = [
        "id", "full_name", "email", "contact_number", 
        "address", "registration_date", "status", "balance"
    ]
    
    print(f"{Fore.CYAN + Style.BRIGHT}🚀 Running Smart Data Mapping...{Style.RESET_ALL}")
    mappings = mapper.suggest_mappings(source_schema, target_schema)
    
    # Display in different beautiful formats
    print(f"\n{Back.MAGENTA + Fore.WHITE + Style.BRIGHT} FORMAT 1: DETAILED BEAUTIFUL VIEW {Style.RESET_ALL}")
    display.display_beautiful_mappings(mappings, source_schema, target_schema)
    
    print(f"\n{Back.GREEN + Fore.WHITE + Style.BRIGHT} FORMAT 2: COMPACT BEAUTIFUL VIEW {Style.RESET_ALL}")
    display.display_compact_beautiful(mappings, source_schema, target_schema)
    
    print(f"\n{Back.BLUE + Fore.WHITE + Style.BRIGHT} FORMAT 3: MAPPING CARDS VIEW {Style.RESET_ALL}")
    display.display_mapping_cards(mappings, source_schema, target_schema)

if __name__ == "__main__":
    demo_beautiful_display()