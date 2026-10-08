"""Isolated literal A^n candidates and a log control with matching bounds."""
from dataclasses import replace
import numpy as np
from .accommodation import PowerResponseModel, ResponseDefinition


class LiteralPowerModel(PowerResponseModel):
    def __init__(self, exponent, beta=None, accommodation_min=.01, *, log_control=False):
        if isinstance(exponent,(bool,np.bool_)) or not np.isfinite(exponent):
            raise ValueError("Declare a finite response exponent")
        if (log_control and exponent!=0) or (not log_control and exponent<=0):
            raise ValueError("Literal powers require n>0; the log control requires n=0")
        if not np.isfinite(accommodation_min) or not 0<accommodation_min<6:
            raise ValueError("Declare a positive accommodation lower bound below 6 D")
        self.response_basis="shifted_boxcox" if log_control else "literal_power"
        self._response=replace(ResponseDefinition(float(exponent)),
            family="literal_power_response_v1",basis=self.response_basis)
        self.lower=np.array([-20.,float(accommodation_min)])
        self.upper=np.array([20.,6.])
        self.name="matched_log" if log_control else f"literal_n{float(exponent):.8g}"
        self.beta=np.zeros(27) if beta is None else np.asarray(beta,float).copy()
        if self.beta.shape!=(27,) or not np.isfinite(self.beta).all():
            raise ValueError("Declare 27 finite coefficients")

    def metadata(self):
        return dict(super().metadata(),bounds_physical=[self.lower.tolist(),self.upper.tolist()],
                    coefficient_count=27,literal_response="(A / 1 D)^n" if self.exponent else "log1p(A / 1 D)")

    @classmethod
    def from_object(cls, obj, allow_failed=False):
        if obj.get("schema")!="literal_power_model_v1":
            raise ValueError("Explicit literal-power schema required")
        meta=obj["accommodation_response"]
        model=cls(meta["exponent"],obj["coefficients"],meta["bounds_physical"][0][1],
                  log_control=meta["basis"]=="shifted_boxcox")
        if meta!=model.metadata() or obj.get("model")!=model.name:
            raise ValueError("Response definition or model identity mismatch")
        if not allow_failed and obj["calibration"]["converged"] is not True:
            raise ValueError("Uncertified fit is not an inference model")
        return model
