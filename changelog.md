# Changelog

## Unreleased

### Added
- Isolated `biomarker-service` for vascular biomarker extraction.
- VascX-ready biomarker architecture docs and patient-facing biomarker review UI.
- Patient biomarker panel with thresholds, bilateral comparison, trend charts, and stats.

### Changed
- Backend now sends raw image bytes to biomarker-service using multipart `application/octet-stream` transport.
- Biomarker-service now validates uploads by decoding bytes instead of rejecting on MIME type.
- Backend prediction flow gates XAI on biomarker completion and persists biomarker metadata.
- README and agent guidance now reflect the biomarker workflow, VascX direction, and updated service layout.
