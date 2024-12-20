function createPredictiveMaintenancePresentation() {
  // Create a new presentation
  var presentation = SlidesApp.create('Predictive Maintenance System');
  
  // Set some constants for styling
  const TITLE_FONT_SIZE = 24;
  const BODY_FONT_SIZE = 14;
  const ACCENT_COLOR = '#1a73e8';
  const BACKGROUND_COLOR = '#ffffff';
  
  // Slide 1: Title
  var titleSlide = presentation.getSlides()[0];
  titleSlide.getBackground().setSolidFill(BACKGROUND_COLOR);
  
  var titleShape = titleSlide.getShapes()[0];
  titleShape.getText()
    .setText('Predictive Maintenance System')
    .getTextStyle()
    .setFontSize(36)
    .setForegroundColor(ACCENT_COLOR)
    .setBold(true);
  
  var subtitleShape = titleSlide.getShapes()[1];
  subtitleShape.getText()
    .setText('Using Machine Learning for Early Failure Detection')
    .getTextStyle()
    .setFontSize(20)
    .setForegroundColor('#666666');
  
  // Slide 2: Executive Summary
  var summarySlide = presentation.appendSlide(SlidesApp.PredefinedLayout.TITLE_AND_BODY);
  summarySlide.getBackground().setSolidFill(BACKGROUND_COLOR);
  
  summarySlide.getShapes()[0].getText()
    .setText('Executive Summary')
    .getTextStyle()
    .setFontSize(TITLE_FONT_SIZE)
    .setForegroundColor(ACCENT_COLOR);
  
  summarySlide.getShapes()[1].getText()
    .setText('• Objective: Predict machine failures before they occur\n' +
             '• Impact: Reduce costs and downtime\n' +
             '• Results: 98.66% accuracy in predictions\n' +
             '• Implementation: Web-based system with REST API\n' +
             '• Status: Deployed and operational')
    .getTextStyle()
    .setFontSize(BODY_FONT_SIZE);
  
  // Slide 3: Problem Statement
  var problemSlide = presentation.appendSlide(SlidesApp.PredefinedLayout.TITLE_AND_TWO_COLUMNS);
  problemSlide.getBackground().setSolidFill(BACKGROUND_COLOR);
  
  problemSlide.getShapes()[0].getText()
    .setText('The Problem')
    .getTextStyle()
    .setFontSize(TITLE_FONT_SIZE)
    .setForegroundColor(ACCENT_COLOR);
  
  problemSlide.getShapes()[1].getText()
    .setText('Current Challenges:\n\n' +
             '• Reactive maintenance costs\n' +
             '• Unplanned downtime\n' +
             '• Production losses\n' +
             '• Premature equipment wear')
    .getTextStyle()
    .setFontSize(BODY_FONT_SIZE);
  
  problemSlide.getShapes()[2].getText()
    .setText('Business Impact:\n\n' +
             '• High repair costs\n' +
             '• Reduced productivity\n' +
             '• Quality issues\n' +
             '• Safety concerns')
    .getTextStyle()
    .setFontSize(BODY_FONT_SIZE);
  
  // Slide 4: Dataset
  var datasetSlide = presentation.appendSlide(SlidesApp.PredefinedLayout.TITLE_AND_BODY);
  datasetSlide.getBackground().setSolidFill(BACKGROUND_COLOR);
  
  datasetSlide.getShapes()[0].getText()
    .setText('Dataset Overview')
    .getTextStyle()
    .setFontSize(TITLE_FONT_SIZE)
    .setForegroundColor(ACCENT_COLOR);
  
  datasetSlide.getShapes()[1].getText()
    .setText('AI4I 2020 Predictive Maintenance Dataset:\n\n' +
             '• 10,000 sensor records\n' +
             '• Key variables monitored:\n' +
             '  - Air temperature\n' +
             '  - Process temperature\n' +
             '  - Rotational speed\n' +
             '  - Torque\n' +
             '  - Tool wear')
    .getTextStyle()
    .setFontSize(BODY_FONT_SIZE);
  
  // Slide 5: Methodology
  var methodSlide = presentation.appendSlide(SlidesApp.PredefinedLayout.TITLE_AND_TWO_COLUMNS);
  methodSlide.getBackground().setSolidFill(BACKGROUND_COLOR);
  
  methodSlide.getShapes()[0].getText()
    .setText('Methodology')
    .getTextStyle()
    .setFontSize(TITLE_FONT_SIZE)
    .setForegroundColor(ACCENT_COLOR);
  
  methodSlide.getShapes()[1].getText()
    .setText('Feature Engineering:\n\n' +
             '• Temperature difference\n' +
             '• Power calculation\n' +
             '• Rolling averages\n' +
             '• Wear rate analysis')
    .getTextStyle()
    .setFontSize(BODY_FONT_SIZE);
  
  methodSlide.getShapes()[2].getText()
    .setText('Model Selection:\n\n' +
             '• Random Forest\n' +
             '• Gradient Boosting\n' +
             '• Neural Networks\n' +
             '• Hyperparameter tuning')
    .getTextStyle()
    .setFontSize(BODY_FONT_SIZE);
  
  // Slide 6: Results
  var resultsSlide = presentation.appendSlide(SlidesApp.PredefinedLayout.TITLE_AND_BODY);
  resultsSlide.getBackground().setSolidFill(BACKGROUND_COLOR);
  
  resultsSlide.getShapes()[0].getText()
    .setText('Results')
    .getTextStyle()
    .setFontSize(TITLE_FONT_SIZE)
    .setForegroundColor(ACCENT_COLOR);
  
  resultsSlide.getShapes()[1].getText()
    .setText('Model Performance:\n\n' +
             '• Overall Accuracy: 98.66%\n' +
             '• F1 Score: 0.988\n' +
             '• Precision (Failure): 0.71\n' +
             '• Recall (Failure): 0.79\n\n' +
             'Key Achievements:\n' +
             '• Early failure detection\n' +
             '• Low false alarm rate\n' +
             '• Real-time predictions')
    .getTextStyle()
    .setFontSize(BODY_FONT_SIZE);
  
  // Slide 7: Implementation
  var implSlide = presentation.appendSlide(SlidesApp.PredefinedLayout.TITLE_AND_BODY);
  implSlide.getBackground().setSolidFill(BACKGROUND_COLOR);
  
  implSlide.getShapes()[0].getText()
    .setText('Implementation')
    .getTextStyle()
    .setFontSize(TITLE_FONT_SIZE)
    .setForegroundColor(ACCENT_COLOR);
  
  implSlide.getShapes()[1].getText()
    .setText('System Components:\n\n' +
             '• Flask REST API\n' +
             '• Machine Learning Model\n' +
             '• Web Interface\n' +
             '• Real-time Predictions\n\n' +
             'Deployment:\n' +
             '• Cloud hosting on Render\n' +
             '• Continuous Integration\n' +
             '• Automated Testing')
    .getTextStyle()
    .setFontSize(BODY_FONT_SIZE);
  
  // Slide 8: Demo
  var demoSlide = presentation.appendSlide(SlidesApp.PredefinedLayout.TITLE_AND_BODY);
  demoSlide.getBackground().setSolidFill(BACKGROUND_COLOR);
  
  demoSlide.getShapes()[0].getText()
    .setText('Live Demo')
    .getTextStyle()
    .setFontSize(TITLE_FONT_SIZE)
    .setForegroundColor(ACCENT_COLOR);
  
  demoSlide.getShapes()[1].getText()
    .setText('Access the live system at:\n\n' +
             'https://predictive-maintenance-vgzh.onrender.com/\n\n' +
             'Features:\n' +
             '• Real-time predictions\n' +
             '• Parameter monitoring\n' +
             '• Failure probability estimation\n' +
             '• Health indicators')
    .getTextStyle()
    .setFontSize(BODY_FONT_SIZE);
  
  // Slide 9: Future Improvements
  var futureSlide = presentation.appendSlide(SlidesApp.PredefinedLayout.TITLE_AND_BODY);
  futureSlide.getBackground().setSolidFill(BACKGROUND_COLOR);
  
  futureSlide.getShapes()[0].getText()
    .setText('Future Improvements')
    .getTextStyle()
    .setFontSize(TITLE_FONT_SIZE)
    .setForegroundColor(ACCENT_COLOR);
  
  futureSlide.getShapes()[1].getText()
    .setText('Planned Enhancements:\n\n' +
             '• Real-time monitoring integration\n' +
             '• Mobile application development\n' +
             '• Advanced analytics dashboard\n' +
             '• Automated maintenance scheduling\n' +
             '• Multi-machine monitoring\n' +
             '• Enhanced visualization tools')
    .getTextStyle()
    .setFontSize(BODY_FONT_SIZE);
  
  // Slide 10: Thank You
  var thanksSlide = presentation.appendSlide(SlidesApp.PredefinedLayout.TITLE_ONLY);
  thanksSlide.getBackground().setSolidFill(BACKGROUND_COLOR);
  
  thanksSlide.getShapes()[0].getText()
    .setText('Thank You!')
    .getTextStyle()
    .setFontSize(36)
    .setForegroundColor(ACCENT_COLOR)
    .setBold(true);
  
  // Return the presentation URL
  return presentation.getUrl();
} 