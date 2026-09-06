"""Deterministic numerical compliance validation."""

from dataclasses import dataclass

from langchain_core.tools import StructuredTool


@dataclass
class ValidationResult:
    """Result of deterministic parameter validation."""

    parameter: str
    value: float
    compliant: bool
    lower_bound: float | None
    upper_bound: float | None
    message: str


class ValidateParameterTool:
    """Validate a numeric value against inclusive compliance bounds in Python."""

    name = "validate_parameter"

    description = (
        "Validate a numeric experimental parameter against the acceptable "
        "lower and upper compliance bounds."
    )

    def run(
        self,
        parameter: str,
        value: float,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ) -> ValidationResult:
        """Return a compliance decision, treating supplied bounds as inclusive."""
        if lower_bound is None and upper_bound is None:
            raise ValueError("At least one compliance bound must be provided.")
        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound > upper_bound
        ):
            raise ValueError("Lower bound cannot be greater than upper bound.")
        if lower_bound is not None and value < lower_bound:
            return ValidationResult(
                parameter, value, False, lower_bound, upper_bound,
                f"{parameter} value {value} is below the lower limit of {lower_bound}.",
            )
        if upper_bound is not None and value > upper_bound:
            return ValidationResult(
                parameter, value, False, lower_bound, upper_bound,
                f"{parameter} value {value} is above the upper limit of {upper_bound}.",
            )
        return ValidationResult(
            parameter, value, True, lower_bound, upper_bound,
            f"{parameter} value {value} is within the acceptable range.",
        )


_validation_tool = ValidateParameterTool()


def _validate_parameter(
    parameter: str,
    value: float,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> dict[str, object]:
    """Adapt deterministic validation to a serialization-friendly tool."""
    result = _validation_tool.run(
        parameter=parameter,
        value=value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    return {
        "parameter": result.parameter,
        "value": result.value,
        "compliant": result.compliant,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "message": result.message,
    }


validate_parameter = StructuredTool.from_function(
    func=_validate_parameter,
    name=_validation_tool.name,
    description=_validation_tool.description,
)
