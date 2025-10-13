
public class LangermannIndFactory implements IndFactory {

    private final int n;

    public LangermannIndFactory(int n) {
        this.n = n;
    }

    @Override
    public LangermannInd getInstance() {
        return new LangermannInd(this.n);
    }
}
