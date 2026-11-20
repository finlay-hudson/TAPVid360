import pytest
import torch

from src.tapvid360.common.conversions.convert import Conversions
from src.tapvid360.common.conversions.mapper_config import MapperConfig


# Dummy configuration for tests
@pytest.fixture
def config():
    return MapperConfig(
        f_x=1000.0,
        f_y=1000.0,
        crop_width=640,
        crop_height=480,
        equirectangular_height=512,
        equirectangular_width=1024,
        fov_x=None,
        fov_y=None,
    )


# Dummy rotation matrix (identity) for tests where R_w2c is required
@pytest.fixture
def R_w2c():
    return torch.eye(3)


# A small tensor input example for each coordinate space
@pytest.fixture
def dummy_ij():
    return torch.tensor([[320.0, 240.0]])


@pytest.fixture
def dummy_cr_centre():
    return torch.tensor([[512.0, 256.0]])


@pytest.fixture
def dummy_cr():
    return torch.tensor([[128.0, 256.0]])


@pytest.fixture
def dummy_thetaphi():
    # theta, phi in radians (within valid ranges)
    return torch.tensor([[1.0, 0.5]])


@pytest.fixture
def dummy_uv():
    return torch.tensor([[0.5, 0.25]])


@pytest.fixture
def dummy_vc():
    return torch.tensor([[1.0, 0.0, 0.0]])


@pytest.fixture
def dummy_vw():
    return torch.tensor([[1.0, 0.0, 0.0]])


# Generic test helper: verify all conversions produce tensors and no error
def run_basic_conversion_tests(mapper, inputs, R_w2c=None):
    # to_cr
    if R_w2c is not None:
        out_cr = mapper.to_cr(inputs, R_w2c)
    else:
        out_cr = mapper.to_cr(inputs)
    assert isinstance(out_cr, torch.Tensor)

    # to_uv
    if R_w2c is not None:
        out_uv = mapper.to_uv(inputs, R_w2c)
    else:
        out_uv = mapper.to_uv(inputs)
    assert isinstance(out_uv, torch.Tensor)

    # to_thetaphi
    if R_w2c is not None:
        out_tp = mapper.to_thetaphi(inputs, R_w2c)
    else:
        out_tp = mapper.to_thetaphi(inputs)
    assert isinstance(out_tp, torch.Tensor)

    # to_vw
    if R_w2c is not None:
        out_vw = mapper.to_vw(inputs, R_w2c)
    else:
        out_vw = mapper.to_vw(inputs)
    assert isinstance(out_vw, torch.Tensor)

    # to_vc
    if R_w2c is not None:
        out_vc = mapper.to_vc(inputs, R_w2c)
    else:
        out_vc = mapper.to_vc(inputs)
    assert isinstance(out_vc, torch.Tensor)

    # to_ij
    if R_w2c is not None:
        out_ij = mapper.to_ij(inputs, R_w2c)
    else:
        out_ij = mapper.to_ij(inputs)
    assert isinstance(out_ij, torch.Tensor)


def test_ConversionIJ(config, dummy_ij, R_w2c):
    mapper = Conversions(config).ij
    run_basic_conversion_tests(mapper, dummy_ij, R_w2c)


def test_ConversionCR(config, dummy_cr, R_w2c):
    mapper = Conversions(config).cr
    run_basic_conversion_tests(mapper, dummy_cr, R_w2c)


def test_ConversionThetaPhi(config, dummy_thetaphi, R_w2c):
    mapper = Conversions(config).thetaphi
    run_basic_conversion_tests(mapper, dummy_thetaphi, R_w2c)


def test_ConversionUV(config, dummy_uv, R_w2c):
    mapper = Conversions(config).uv
    run_basic_conversion_tests(mapper, dummy_uv, R_w2c)


def test_ConversionVc(config, dummy_vc, R_w2c):
    mapper = Conversions(config).vc
    run_basic_conversion_tests(mapper, dummy_vc, R_w2c)


def test_ConversionVw(config, dummy_vw, R_w2c):
    mapper = Conversions(config).vw
    run_basic_conversion_tests(mapper, dummy_vw, R_w2c)


def test_round_trip_IJ_VC_IJ(config, dummy_ij, R_w2c):
    converter = Conversions(config)
    vc = converter.ij.to_vc(dummy_ij, R_w2c)
    ij = converter.vc.to_ij(vc, R_w2c)
    assert torch.allclose(dummy_ij, ij, atol=1e-2), "Round-trip IJ -> VC -> IJ failed"


def test_round_trip_CR_IJ_CR_full_steps(config, dummy_cr_centre, R_w2c):
    converter = Conversions(config)
    uv_ = converter.cr.to_uv(dummy_cr_centre)
    thetaphi_ = converter.uv.to_thetaphi(uv_)
    vw_ = converter.thetaphi.to_vw(thetaphi_)
    vc_ = converter.vw.to_vc(vw_, R_w2c)
    ij_ = converter.vc.to_ij(vc_, R_w2c)

    vc = converter.ij.to_vc(ij_, R_w2c)
    vw = converter.vc.to_vw(vc, R_w2c)
    thetaphi = converter.vw.to_thetaphi(vw, R_w2c)
    uv = converter.thetaphi.to_uv(thetaphi, R_w2c)
    cr = converter.uv.to_cr(uv, R_w2c)

    assert torch.allclose(cr, dummy_cr_centre, atol=1e-2), "Round-trip CR -> IJ -> CR failed"


def test_round_trip_CR_UV_CR(config, dummy_cr, R_w2c):
    converter = Conversions(config)
    uv_ = converter.cr.to_uv(dummy_cr)
    cr = converter.uv.to_cr(uv_, R_w2c)

    assert torch.allclose(dummy_cr, cr, atol=1e-2), "Round-trip CR -> UV -> CR failed"


def test_round_trip_CR_THETAPHI_CR(config, dummy_cr, R_w2c):
    converter = Conversions(config)
    thetaphi = converter.cr.to_thetaphi(dummy_cr)
    cr = converter.thetaphi.to_cr(thetaphi, R_w2c)

    assert torch.allclose(dummy_cr, cr, atol=1e-2), "Round-trip CR -> THETAPHI -> CR failed"


def test_round_trip_CR_VW_CR(config, dummy_cr, R_w2c):
    converter = Conversions(config)
    vw = converter.cr.to_vw(dummy_cr)
    cr = converter.vw.to_cr(vw, R_w2c)

    assert torch.allclose(dummy_cr, cr, atol=1e-2), "Round-trip CR -> VW -> CR failed"


def test_round_trip_CR_VC_CR(config, dummy_cr, R_w2c):
    converter = Conversions(config)
    vc = converter.cr.to_vc(dummy_cr, R_w2c)
    cr = converter.vc.to_cr(vc, R_w2c)

    assert torch.allclose(dummy_cr, cr, atol=1e-2), "Round-trip CR -> VC -> CR failed"


def test_round_trip_CR_IJ_CR(config, dummy_cr_centre, R_w2c):
    converter = Conversions(config)
    ij = converter.cr.to_ij(dummy_cr_centre, R_w2c)
    cr = converter.ij.to_cr(ij, R_w2c)

    assert torch.allclose(dummy_cr_centre, cr, atol=1e-2), "Round-trip CR -> IJ -> CR failed"


@pytest.mark.xfail
def test_round_trip_CR_IJ_CR_uncentred(config, dummy_cr, R_w2c):
    # This test fails as the CR with the identity rotation leads to the pixel being outside of the frame. So we need
    # guardrails around that # TODO
    converter = Conversions(config)
    ij = converter.cr.to_ij(dummy_cr, R_w2c)
    cr = converter.ij.to_cr(ij, R_w2c)

    assert torch.allclose(dummy_cr, cr, atol=1e-2), "Round-trip CR -> IJ -> CR failed"


def test_round_trip_VC_IJ_VC(config, dummy_vc, R_w2c):
    converter = Conversions(config)
    ij = converter.vc.to_ij(dummy_vc, R_w2c)
    vc = converter.ij.to_vc(ij, R_w2c)

    assert torch.allclose(dummy_vc, vc, atol=1e-2), "Round-trip VC -> IJ -> VC failed"
