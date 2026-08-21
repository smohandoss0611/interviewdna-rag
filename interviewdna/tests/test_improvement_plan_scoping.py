from services.evaluation_service import _filter_alignment_to_covered


def test_filter_drops_jd_requirements_never_covered_in_interview():
    """Regression test: the improvement plan previously received the FULL
    resume<->JD alignment, including requirements that were never part of
    the actual interview (e.g. education, unassessed soft skills). A model
    would happily echo those clean JD bullets back as "development areas"
    even though nothing about them was ever tested. This locks in that only
    competencies actually touched (TESTED/PARTIAL) survive filtering."""
    alignment = {
        "items": [
            {"requirement": "Strong SQL and Python skills", "evidence_level": "STRONG_EVIDENCE"},
            {"requirement": "Experience with Kubernetes", "evidence_level": "NOT_DEMONSTRATED"},
            {"requirement": "Bachelor's or advanced degree in Computer Science", "evidence_level": "STRONG_EVIDENCE"},
            {"requirement": "Excellent written and verbal communication skills", "evidence_level": "PARTIAL_EVIDENCE"},
        ]
    }
    # Only "Strong SQL and Python skills" was actually asked about this session.
    coverage = {
        "Strong SQL and Python skills": "TESTED",
        "Kubernetes": "NOT_TESTED",
    }

    result = _filter_alignment_to_covered(alignment, coverage)
    requirements = [item["requirement"] for item in result["items"]]

    assert "Strong SQL and Python skills" in requirements
    assert "Experience with Kubernetes" not in requirements  # NOT_TESTED -> dropped
    assert not any("degree" in r.lower() for r in requirements)  # never in coverage at all
    assert not any("communication" in r.lower() for r in requirements)  # never in coverage at all
    assert len(requirements) == 1


def test_filter_keeps_partial_coverage_not_just_tested():
    """A competency that got SOME turns (PARTIAL) but wasn't fully resolved
    should still be eligible -- something was actually observed about it."""
    alignment = {"items": [{"requirement": "Streaming data systems (Kafka)", "evidence_level": "PARTIAL_EVIDENCE"}]}
    coverage = {"Streaming data systems (Kafka)": "PARTIAL"}

    result = _filter_alignment_to_covered(alignment, coverage)
    assert len(result["items"]) == 1


def test_filter_returns_empty_when_nothing_covered():
    alignment = {"items": [{"requirement": "Anything", "evidence_level": "STRONG_EVIDENCE"}]}
    result = _filter_alignment_to_covered(alignment, coverage={})
    assert result == {"items": []}
