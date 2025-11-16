# AI Persona Usability Testing

AI personas simulate real user behavior to test the Coterie website (https://www.coterie.com/).

## Setup & Run

```bash
# Run setup script (activate bluepill conda env + install dependencies)
bash setup.sh

# Test all 5 personas
python main.py

# Test single persona
python main.py --single sarah_kim

# Run in headless mode
python main.py --headless
```

Available personas: `sarah_kim`, `maya_rodriguez`, `lauren_peterson`, `jasmine_lee`, `priya_desai`

## Notes

- **Pre-test login & popup handling**: The system logs in and closes popups before testing to simulate authentic human user paths, as real users wouldn't trigger anti-bot systems.
- **Behavioral variations**: Parameters like `max_scrolls_per_page` may cause different persona behaviors across runs.
- **Demo variability**: Each run produces slightly different paths and decisions—demo outputs are reference only.