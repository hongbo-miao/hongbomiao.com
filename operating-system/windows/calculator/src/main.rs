use windows::core::{s, Interface, Result};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED,
};
use windows::Win32::UI::Accessibility::{CUIAutomation, IUIAutomation, IUIAutomationElement};
use windows::Win32::UI::WindowsAndMessaging::FindWindowA;
use windows::UI::UIAutomation::AutomationElement;

fn main() -> Result<()> {
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
        let window = FindWindowA(None, s!("Calculator"))?;

        // Start with COM API
        let automation: IUIAutomation = CoCreateInstance(&CUIAutomation, None, CLSCTX_ALL)?;
        let element: IUIAutomationElement = automation.ElementFromHandle(window)?;

        // Use COM API
        let name = element.CurrentName()?;
        println!("Window name: {}", name);

        // Query for WinRT API (will fail on earlier versions of Windows)
        let element: Result<AutomationElement> = element.cast();

        if let Ok(element) = element {
            // Use WinRT API
            println!("File name: {}", element.ExecutableFileName()?);
        }
    }

    Ok(())
}
