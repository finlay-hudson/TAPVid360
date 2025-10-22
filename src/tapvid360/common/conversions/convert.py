import inspect
import torch

from src.tapvid360.common.conversions.mapper_atomics import (cr_to_uv, uv_to_cr, uv_to_thetaphi, thetaphi_to_uv,
                                                             thetaphi_to_vw, vw_to_thetaphi, vw_to_vc, vc_to_vw,
                                                             vc_to_ij, ij_to_vc)
from src.tapvid360.common.conversions.mapper_config import MapperConfig


class SingleConversionBase:
    def __init__(self, config: MapperConfig):
        self.cfg = config
        self.chain = {}

    def _run(self, target: str, var: torch.Tensor, R_w2c: torch.Tensor = None):
        if target not in self.chain:
            raise NotImplementedError(f"Conversion to {target} not implemented.")
        func = self.chain[target]
        params = inspect.signature(func).parameters
        if 'R' in params:
            return func(var, R=R_w2c)
        else:
            return func(var)

    def to_cr(self, var: torch.Tensor, R_w2c: torch.Tensor = None):
        return self._run('cr', var, R_w2c)

    def to_uv(self, var: torch.Tensor, R_w2c: torch.Tensor = None):
        return self._run('uv', var, R_w2c)

    def to_thetaphi(self, var: torch.Tensor, R_w2c: torch.Tensor = None):
        return self._run('thetaphi', var, R_w2c)

    def to_vw(self, var: torch.Tensor, R_w2c: torch.Tensor = None):
        return self._run('vw', var, R_w2c)

    def to_vc(self, var: torch.Tensor, R_w2c: torch.Tensor = None):
        return self._run('vc', var, R_w2c)

    def to_ij(self, var: torch.Tensor, R_w2c: torch.Tensor = None):
        return self._run('ij', var, R_w2c)


class Conversions:
    def __init__(self, config: MapperConfig):
        self.config = config
        self.cr = ConversionCR(config)
        self.uv = ConversionUV(config)
        self.thetaphi = ConversionThetaPhi(config)
        self.vw = ConversionVw(config)
        self.vc = ConversionVc(config)
        self.ij = ConversionIJ(config)


class ConversionIJ(SingleConversionBase):
    def __init__(self, config: MapperConfig):
        super().__init__(config)
        self.chain = {
            'ij': lambda ij: ij,
            'vc': lambda ij: ij_to_vc(ij, self.cfg.f_x, self.cfg.f_y, self.cfg.crop_width, self.cfg.crop_height),
            'vw': lambda ij, R: vc_to_vw(
                ij_to_vc(ij, self.cfg.f_x, self.cfg.f_y, self.cfg.crop_width, self.cfg.crop_height), R),
            'thetaphi': lambda ij, R: vw_to_thetaphi(
                vc_to_vw(ij_to_vc(ij, self.cfg.f_x, self.cfg.f_y, self.cfg.crop_width, self.cfg.crop_height), R)
            ),
            'uv': lambda ij, R: thetaphi_to_uv(
                vw_to_thetaphi(
                    vc_to_vw(ij_to_vc(ij, self.cfg.f_x, self.cfg.f_y, self.cfg.crop_width, self.cfg.crop_height), R)
                )
            ),
            'cr': lambda ij, R: uv_to_cr(
                thetaphi_to_uv(
                    vw_to_thetaphi(
                        vc_to_vw(ij_to_vc(ij, self.cfg.f_x, self.cfg.f_y, self.cfg.crop_width, self.cfg.crop_height), R)
                    )
                ),
                self.cfg.equirectangular_height
            )
        }


class ConversionCR(SingleConversionBase):
    def __init__(self, config: MapperConfig):
        super().__init__(config)
        self.chain = {
            'cr': lambda cr: cr,
            'uv': lambda cr: cr_to_uv(cr, self.cfg.equirectangular_height),
            'thetaphi': lambda cr: uv_to_thetaphi(cr_to_uv(cr, self.cfg.equirectangular_height)),
            'vw': lambda cr: thetaphi_to_vw(uv_to_thetaphi(cr_to_uv(cr, self.cfg.equirectangular_height))),
            'vc': lambda cr, R: vw_to_vc(
                thetaphi_to_vw(uv_to_thetaphi(cr_to_uv(cr, self.cfg.equirectangular_height))), R),
            'ij': lambda cr, R: vc_to_ij(
                vw_to_vc(thetaphi_to_vw(uv_to_thetaphi(cr_to_uv(cr, self.cfg.equirectangular_height))), R),
                self.cfg.f_x, self.cfg.f_y, self.cfg.crop_width, self.cfg.crop_height
            )
        }


class ConversionThetaPhi(SingleConversionBase):
    def __init__(self, config: MapperConfig):
        super().__init__(config)
        self.chain = {
            'thetaphi': lambda tp: tp,
            'uv': lambda tp: thetaphi_to_uv(tp),
            'cr': lambda tp: uv_to_cr(thetaphi_to_uv(tp), self.cfg.equirectangular_height),
            'vw': lambda tp: thetaphi_to_vw(tp),
            'vc': lambda tp, R: vw_to_vc(thetaphi_to_vw(tp), R),
            'ij': lambda tp, R: vc_to_ij(
                vw_to_vc(thetaphi_to_vw(tp), R),
                self.cfg.f_x, self.cfg.f_y, self.cfg.crop_width, self.cfg.crop_height
            )
        }


class ConversionUV(SingleConversionBase):
    def __init__(self, config: MapperConfig):
        super().__init__(config)
        self.chain = {
            'uv': lambda uv: uv,
            'cr': lambda uv: uv_to_cr(uv, self.cfg.equirectangular_height),
            'thetaphi': lambda uv: uv_to_thetaphi(uv),
            'vw': lambda uv: thetaphi_to_vw(uv_to_thetaphi(uv)),
            'vc': lambda uv, R: vw_to_vc(thetaphi_to_vw(uv_to_thetaphi(uv)), R),
            'ij': lambda uv, R: vc_to_ij(
                vw_to_vc(thetaphi_to_vw(uv_to_thetaphi(uv)), R),
                self.cfg.f_x, self.cfg.f_y, self.cfg.crop_width, self.cfg.crop_height
            )
        }


class ConversionVc(SingleConversionBase):
    def __init__(self, config: MapperConfig):
        super().__init__(config)
        self.chain = {
            'vc': lambda vc: vc,
            'vw': lambda vc, R: vc_to_vw(vc, R),
            'thetaphi': lambda vc, R: vw_to_thetaphi(vc_to_vw(vc, R)),
            'uv': lambda vc, R: thetaphi_to_uv(vw_to_thetaphi(vc_to_vw(vc, R))),
            'cr': lambda vc, R: uv_to_cr(thetaphi_to_uv(vw_to_thetaphi(vc_to_vw(vc, R))),
                                         self.cfg.equirectangular_height),
            'ij': lambda vc: vc_to_ij(vc, self.cfg.f_x, self.cfg.f_y, self.cfg.crop_width, self.cfg.crop_height)
        }


class ConversionVw(SingleConversionBase):
    def __init__(self, config: MapperConfig):
        super().__init__(config)
        self.chain = {
            'vw': lambda vw: vw,
            'thetaphi': lambda vw: vw_to_thetaphi(vw),
            'uv': lambda vw: thetaphi_to_uv(vw_to_thetaphi(vw)),
            'cr': lambda vw: uv_to_cr(thetaphi_to_uv(vw_to_thetaphi(vw)), self.cfg.equirectangular_height),
            'vc': lambda vw, R: vw_to_vc(vw, R),
            'ij': lambda vw, R: vc_to_ij(vw_to_vc(vw, R), self.cfg.f_x, self.cfg.f_y, self.cfg.crop_width,
                                         self.cfg.crop_height)
        }
