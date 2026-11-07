from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

out = Path('output/pdf/ethachi_xgboost_problemi.pdf')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom', fontName='Helvetica-Bold', fontSize=23, leading=28, textColor=HexColor('#15324b'), spaceAfter=9))
styles.add(ParagraphStyle(name='SubtitleCustom', fontName='Helvetica', fontSize=10, leading=15, textColor=HexColor('#536474'), spaceAfter=20))
styles.add(ParagraphStyle(name='SectionCustom', fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=HexColor('#15324b'), spaceAfter=7))
styles.add(ParagraphStyle(name='BodyCustom', fontName='Helvetica', fontSize=10.5, leading=15, spaceAfter=7))
styles.add(ParagraphStyle(name='NoteCustom', fontName='Helvetica', fontSize=9, leading=13, textColor=HexColor('#536474'), spaceAfter=7))
P=lambda text: Paragraph(text, styles['BodyCustom'])
story = [Paragraph('XGBoost: integration issues', styles['TitleCustom']), Paragraph('ETHachi-TSFM | Technical brief for the team | 7 November 2026', styles['SubtitleCustom']), P('<b>Scope.</b> Three issues in the code integrating XGBoost into the dashboard and command-line benchmark. These are not defects in the XGBoost library and do not directly affect the three foundation models.'), Spacer(1,10)]
sections = [
('1. Dashboard: train/test splitting', '<b>Location:</b> app.py, XGBoost block (lines 237-239); src/models.py, split_train_test().', '<b>Issue and impact:</b> the caller expects four outputs, but the function returns only the training and test subsets of one DataFrame. XGBoost fails with <font face="Courier">expected 4, got 2</font>.', '<b>Proposed fix:</b> call the function separately for features and targets to obtain X_train, X_test, y_train, and y_test.'),
('2. Dashboard: forecast date alignment', '<b>Location:</b> app.py, selection of the first predictions (line 241).', '<b>Issue and impact:</b> the code always takes the first 156 test predictions, even when a different forecast date is selected. Once issue 1 is fixed, this can lead to comparisons and metrics calculated across different time periods.', '<b>Proposed fix:</b> attach the X_test time index to the predictions and select the timestamps in y_true_horizon. Check that all 156 steps are available and the selected window falls within the test period.'),
('3. Command line: cross-validation', '<b>Location:</b> main.py, split() (lines 70-84) and its XGBoost caller (line 160).', '<b>Issue and impact:</b> when USE_CROSS_VALIDATION=True, split() returns two fold iterators, while the caller expects four datasets. The XGBoost execution path fails. With False, this specific error does not occur.', '<b>Proposed fix:</b> handle cross-validation in a dedicated execution path: align features and targets for each fold, train, evaluate, and aggregate metrics. Keep final test evaluation separate.')]
for title, where, problem, fix in sections:
    story.append(KeepTogether([Paragraph(title, styles['SectionCustom']), Paragraph(where, styles['NoteCustom']), P(problem), P(fix), Spacer(1,10)]))
story += [P('<b>Connection between issues 1 and 3.</b> Both involve a mismatch between caller and function: four values expected, two returned. Issue 1 requires two separate splits; issue 3 also requires a loop over folds.'), Paragraph('<b>Status:</b> issues confirmed through code inspection; issue 1 also observed in the dashboard. Fixes are proposed and have not yet been implemented or validated by execution. Suggested priority: 1, then 2, then 3 if cross-validation is needed.', styles['NoteCustom'])]
def footer(canvas, doc):
    canvas.setStrokeColor(HexColor('#d9e2e9'))
    canvas.line(43,38,A4[0]-43,38)
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(HexColor('#536474'))
    canvas.drawString(43,25,'ETHachi-TSFM | Local code review')
    canvas.drawRightString(A4[0]-43,25,str(doc.page))
SimpleDocTemplate(str(out), pagesize=A4, rightMargin=43,leftMargin=43,topMargin=38,bottomMargin=48).build(story,onFirstPage=footer,onLaterPages=footer)
import fitz
pdf=fitz.open(out)
assert len(pdf)==1, f'Expected one page, found {len(pdf)}'
page=pdf[0]
page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save('tmp/pdfs/ethachi_xgboost_preview.png')
print(f'Created and rendered: {out} ({len(pdf)} page)')
